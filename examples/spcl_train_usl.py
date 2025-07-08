from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import sys
import collections
import copy
import time
from datetime import timedelta
from scipy.spatial.distance import cdist

from sklearn.cluster import DBSCAN

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from spcl import datasets
from spcl import models
from spcl.models.hm import HybridMemory
from spcl.trainers import SpCLTrainer_USL
from spcl.evaluators import Evaluator, extract_features
from spcl.utils.data import IterLoader
from spcl.utils.data import transforms as T
from spcl.utils.data.sampler import RandomMultipleGallerySampler
from spcl.utils.data.preprocessor import Preprocessor
from spcl.utils.logging import Logger
from spcl.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict
from spcl.utils.faiss_rerank import compute_jaccard_distance


start_epoch = best_mAP = 0

def jaccard_rerank(features, k1=20, k2=6, use_cosine=True):
    """
    Pure-Python k-reciprocal + Jaccard re-ranking.
    features: (N, D) np.ndarray
    Returns: (N, N) distance matrix
    """
    # 1) initial distance matrix
    if use_cosine:
        # normalize to unit length, then dot → cosine similarities
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        feats = features / (norms + 1e-12)
        sim = np.dot(feats, feats.T)
        dist = 1.0 - sim
    else:
        dist = cdist(features, features, metric='euclidean')

    N = dist.shape[0]
    initial_rank = np.argsort(dist, axis=1)

    # 2) build k-reciprocal sets
    k_half = k1 // 2
    V = np.zeros_like(dist, dtype=np.float32)

    for i in range(N):
        forward = initial_rank[i, :k1+1]
        backward = initial_rank[forward, :k1+1]
        # mutual neighbours
        recip = forward[np.where(backward == i)[0]]

        # expansion step
        recip_exp = recip.copy()
        for j in recip:
            f2 = initial_rank[j, :k_half+1]
            b2 = initial_rank[f2, :k_half+1]
            recip2 = f2[np.where(b2 == j)[0]]
            if len(np.intersect1d(recip2, recip)) > (2/3)*len(recip2):
                recip_exp = np.unique(np.concatenate([recip_exp, recip2]))

        # soft weights on those neighbours
        d_i = dist[i, recip_exp]
        w = np.exp(-d_i)
        V[i, recip_exp] = w / (w.sum() + 1e-12)

    # 3) query expansion (optional)
    if k2 > 1:
        V_qe = np.zeros_like(V, dtype=np.float32)
        for i in range(N):
            V_qe[i] = V[initial_rank[i, :k2]].mean(axis=0)
        V = V_qe

    # 4) Jaccard distance
    jaccard = np.zeros_like(dist, dtype=np.float32)
    for i in range(N):
        min_ = np.minimum(V[i], V)
        max_ = np.maximum(V[i], V)
        jaccard[i] = 1.0 - (min_.sum(axis=1) / (max_.sum(axis=1) + 1e-12))

    return jaccard

# Wraps spcl.datasets.create and returns a dataset object 
# whose train, query, gallery, images_dir attributes match the Market-1501/DUKE/etc. 
def get_data(name, data_dir):
    root = osp.join(data_dir, name)
    dataset = datasets.create(name, root)
    return dataset


# Builds an IterLoader that loops over a DataLoader forever 
# so each epoch can be defined by a fixed number of iterations (args.iters).
def get_train_loader(args, dataset, height, width, batch_size, workers,
                    num_instances, iters, trainset=None):

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    train_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.RandomHorizontalFlip(p=0.5),
             T.Pad(10),
             T.RandomCrop((height, width)),
             T.ToTensor(),
             normalizer,
	         T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
         ])

    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None
    train_loader = IterLoader(
                DataLoader(Preprocessor(train_set, root=dataset.images_dir, transform=train_transformer),
                            batch_size=batch_size, num_workers=workers, sampler=sampler,
                            shuffle=not rmgs_flag, pin_memory=True, drop_last=False), length=iters)

    return train_loader

# Simple loader (no augmentation) for query, gallery or any arbitrary list of image paths 
# (used later to feed the whole training set through the model for feature extraction).
def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.ToTensor(),
             normalizer
         ])

    if (testset is None):
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader

# Instantiates a backbone (resnet50 by default) with the classification head removed (num_classes=0). 
# The model is wrapped in nn.DataParallel and moved to CUDA.
def create_model(args):
    model = models.create(args.arch, num_features=args.features, norm=True, dropout=args.dropout, num_classes=0)
    # use CUDA
    #model.cuda()
    #model = nn.DataParallel(model)
    model = model.to(args.device)
    return model


def main():
    #args = parser.parse_args()
    parser.add_argument('--device', default='cpu', help="torch device (cpu only)")
    args = parser.parse_args()
    args.device = 'cpu'

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        #cudnn.deterministic = True

    main_worker(args)


def main_worker(args):
    global start_epoch, best_mAP
    start_time = time.monotonic()

    #cudnn.benchmark = False

    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # Create datasets
    iters = args.iters if (args.iters>0) else None
    print("==> Load unlabeled dataset")
    dataset = get_data(args.dataset, args.data_dir)

    import random
    from collections import defaultdict

    # ——— START quick subset by identity ———
    random.seed(42)

    # 1) pick 10 IDs at random
    all_pids     = list({ pid for _, pid, _ in dataset.train })
    sampled_pids = set(random.sample(all_pids, k=10))

    # 2) filter each split to only those IDs
    dataset.train   = [x for x in dataset.train   if x[1] in sampled_pids]
    dataset.query   = [x for x in dataset.query   if x[1] in sampled_pids]
    dataset.gallery = [x for x in dataset.gallery if x[1] in sampled_pids]

    # (optional) if you also want to cap each split to 100 images total:
    dataset.train   = dataset.train[:200]
    dataset.query   = dataset.query[:200]
    dataset.gallery = dataset.gallery[:200]
    # ——— END quick subset ———


    test_loader = get_test_loader(dataset, args.height, args.width, args.batch_size, args.workers)

    # Create model
    model = create_model(args)

    # Create hybrid memory
    #memory = HybridMemory(model.module.num_features, len(dataset.train), temp=args.temp, momentum=args.momentum).cuda()
    memory = HybridMemory(
            model.num_features if not hasattr(model, 'module') else model.module.num_features,
            len(dataset.train),
            temp=args.temp,
            momentum=args.momentum
        ).to(args.device)

    # Initialize target-domain instance features
    print("==> Initialize instance features in the hybrid memory")
    cluster_loader = get_test_loader(dataset, args.height, args.width,
                                    args.batch_size, args.workers, testset=sorted(dataset.train))
    features, _ = extract_features(model, cluster_loader, print_freq=50)
    features = torch.cat([features[f].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0)
    #memory.features = F.normalize(features, dim=1).cuda()
    memory.features = F.normalize(features, dim=1).to(args.device)

    del cluster_loader, features

    # Evaluator
    evaluator = Evaluator(model)

    # Optimizer
    params = [{"params": [value]} for _, value in model.named_parameters() if value.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

    # Trainer
    trainer = SpCLTrainer_USL(model, memory)

    for epoch in range(args.epochs):

        print('==> Create pseudo labels for unlabeled data with self-paced policy')
        features = memory.features.clone().cpu().numpy()

        rerank_dist = jaccard_rerank(
            features,
            k1=args.k1,
            k2=args.k2,
            use_cosine=True     
        )
                
        del features

        if (epoch==0):
            # DBSCAN cluster
            eps = args.eps
            eps_tight = eps-args.eps_gap
            eps_loose = eps+args.eps_gap
            print('Clustering criterion: eps: {:.3f}, eps_tight: {:.3f}, eps_loose: {:.3f}'.format(eps, eps_tight, eps_loose))
            cluster = DBSCAN(eps=eps, min_samples=4, metric='precomputed', n_jobs=-1)
            cluster_tight = DBSCAN(eps=eps_tight, min_samples=4, metric='precomputed', n_jobs=-1)
            cluster_loose = DBSCAN(eps=eps_loose, min_samples=4, metric='precomputed', n_jobs=-1)

        # select & cluster images as training set of this epochs
        pseudo_labels = cluster.fit_predict(rerank_dist)
        pseudo_labels_tight = cluster_tight.fit_predict(rerank_dist)
        pseudo_labels_loose = cluster_loose.fit_predict(rerank_dist)
        num_ids = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)
        num_ids_tight = len(set(pseudo_labels_tight)) - (1 if -1 in pseudo_labels_tight else 0)
        num_ids_loose = len(set(pseudo_labels_loose)) - (1 if -1 in pseudo_labels_loose else 0)

        # generate new dataset and calculate cluster centers
        def generate_pseudo_labels(cluster_id, num):
            labels = []
            outliers = 0
            for i, ((fname, _, cid), id) in enumerate(zip(sorted(dataset.train), cluster_id)):
                if id!=-1:
                    labels.append(id)
                else:
                    labels.append(num+outliers)
                    outliers += 1
            return torch.Tensor(labels).long()

        pseudo_labels = generate_pseudo_labels(pseudo_labels, num_ids)
        pseudo_labels_tight = generate_pseudo_labels(pseudo_labels_tight, num_ids_tight)
        pseudo_labels_loose = generate_pseudo_labels(pseudo_labels_loose, num_ids_loose)

        # compute R_indep and R_comp
        N = pseudo_labels.size(0)
        label_sim = pseudo_labels.expand(N, N).eq(pseudo_labels.expand(N, N).t()).float()
        label_sim_tight = pseudo_labels_tight.expand(N, N).eq(pseudo_labels_tight.expand(N, N).t()).float()
        label_sim_loose = pseudo_labels_loose.expand(N, N).eq(pseudo_labels_loose.expand(N, N).t()).float()

        R_comp = 1-torch.min(label_sim, label_sim_tight).sum(-1)/torch.max(label_sim, label_sim_tight).sum(-1)
        R_indep = 1-torch.min(label_sim, label_sim_loose).sum(-1)/torch.max(label_sim, label_sim_loose).sum(-1)
        assert((R_comp.min()>=0) and (R_comp.max()<=1))
        assert((R_indep.min()>=0) and (R_indep.max()<=1))

        cluster_R_comp, cluster_R_indep = collections.defaultdict(list), collections.defaultdict(list)
        cluster_img_num = collections.defaultdict(int)
        for i, (comp, indep, label) in enumerate(zip(R_comp, R_indep, pseudo_labels)):
            cluster_R_comp[label.item()].append(comp.item())
            cluster_R_indep[label.item()].append(indep.item())
            cluster_img_num[label.item()]+=1

        cluster_R_comp = [min(cluster_R_comp[i]) for i in sorted(cluster_R_comp.keys())]
        cluster_R_indep = [min(cluster_R_indep[i]) for i in sorted(cluster_R_indep.keys())]
        cluster_R_indep_noins = [iou for iou, num in zip(cluster_R_indep, sorted(cluster_img_num.keys())) if cluster_img_num[num]>1]
        if (epoch==0):
            indep_thres = np.sort(cluster_R_indep_noins)[min(len(cluster_R_indep_noins)-1,np.round(len(cluster_R_indep_noins)*0.9).astype('int'))]

        pseudo_labeled_dataset = []
        outliers = 0
        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset.train), pseudo_labels)):
            indep_score = cluster_R_indep[label.item()]
            comp_score = R_comp[i]
            if ((indep_score<=indep_thres) and (comp_score.item()<=cluster_R_comp[label.item()])):
                pseudo_labeled_dataset.append((fname,label.item(),cid))
            else:
                pseudo_labeled_dataset.append((fname,len(cluster_R_indep)+outliers,cid))
                pseudo_labels[i] = len(cluster_R_indep)+outliers
                outliers+=1

        # statistics of clusters and un-clustered instances
        index2label = collections.defaultdict(int)
        for label in pseudo_labels:
            index2label[label.item()]+=1
        index2label = np.fromiter(index2label.values(), dtype=float)
        print('==> Statistics for epoch {}: {} clusters, {} un-clustered instances, R_indep threshold is {}'
                    .format(epoch, (index2label>1).sum(), (index2label==1).sum(), 1-indep_thres))

        #memory.labels = pseudo_labels.cuda()
        memory.labels = pseudo_labels.to(args.device)

        train_loader = get_train_loader(args, dataset, args.height, args.width,
                                            args.batch_size, args.workers, args.num_instances, iters,
                                            trainset=pseudo_labeled_dataset)

        train_loader.new_epoch()

        trainer.train(epoch, train_loader, optimizer,
                    print_freq=args.print_freq, train_iters=len(train_loader))

        if ((epoch+1)%args.eval_step==0 or (epoch==args.epochs-1)):
            mAP = evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=False)
            is_best = (mAP>best_mAP)
            best_mAP = max(mAP, best_mAP)
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
            }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

            print('\n * Finished epoch {:3d}  model mAP: {:5.1%}  best: {:5.1%}{}\n'.
                  format(epoch, mAP, best_mAP, ' *' if is_best else ''))

        lr_scheduler.step()

    print ('==> Test with the best model:')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=True)

    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Self-paced contrastive learning on unsupervised re-ID")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='market1501',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--num-instances', type=int, default=0,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    # cluster
    parser.add_argument('--eps', type=float, default=0.28,
                        help="max neighbor distance for DBSCAN")
    parser.add_argument('--eps-gap', type=float, default=0.00,
                        help="multi-scale criterion for measuring cluster reliability")
    parser.add_argument('--k1', type=int, default=15,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--k2', type=int, default=4,
                        help="hyperparameter for jaccard distance")
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--momentum', type=float, default=0.2,
                        help="update momentum for the hybrid memory")
    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--iters', type=int, default=150)
    parser.add_argument('--step-size', type=int, default=20)
    # training configs
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--eval-step', type=int, default=10)
    parser.add_argument('--temp', type=float, default=0.05,
                        help="temperature for scaling contrastive loss")
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    main()
