# -*- coding: utf-8 -*-
"""
@Time: 2023/4/30 16:01 
@Author: Marigold
@Version: 0.0.0
@Description：
@WeChat Account: Marigold
"""
import torch
import torch.nn.functional as F

from torch.optim import Adam
from sklearn.cluster import KMeans
from module.GAT_for_GCSEE import GAT
from utils.evaluation import eva
from utils.result import Result
from utils.utils import get_format_variables


def train(args, data, logger):
    pretrain_params_dict = {"acm": [50, 2e-4],
                            "cite": [50, 1e-3],
                            "dblp": [50, 1e-3],
                            "cora": [50, 2e-4],
                            "usps": [200, 2e-4],
                            "amap": [50,  5e-5]}
    args.pretrain_epoch = pretrain_params_dict[args.dataset_name][0]
    args.pretrain_lr = pretrain_params_dict[args.dataset_name][1]
    args.embedding_dim = 10
    args.alpha = 0.2
    # args.weight_decay = 5e-3

    pretrain_gat_filename = args.pretrain_save_path + args.dataset_name + ".pkl"
    model = GAT(args.input_dim, args.embedding_dim, 500, 500, 2000, args.alpha).to(args.device)
    logger.info(model)
    optimizer = Adam(model.parameters(), lr=args.pretrain_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.pretrain_epoch)

    M = data.M.to(args.device).float()
    adj = data.adj.to(args.device).float()
    feature = data.feature.to(args.device).float()
    label = data.label

    acc_max, embedding = 0, None
    max_acc_corresponding_metrics = [0, 0, 0, 0]
    for epoch in range(1, args.pretrain_epoch + 1):
        model.train()
        A_pred, embedding = model(feature, adj, M)
        loss = F.binary_cross_entropy(A_pred.view(-1), adj.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            model.eval()
            kmeans = KMeans(n_clusters=args.clusters, n_init=20).fit(embedding.data.cpu().numpy())
            acc, nmi, ari, f1 = eva(label, kmeans.labels_)
            if acc > acc_max:
                acc_max = acc
                max_acc_corresponding_metrics = [acc, nmi, ari, f1]
            logger.info(get_format_variables(epoch=f"{epoch:0>3d}", acc=f"{acc:0>.4f}", nmi=f"{nmi:0>.4f}",
                                             ari=f"{ari:0>.4f}", f1=f"{f1:0>.4f}"))
    torch.save(model.state_dict(), pretrain_gat_filename)
    result = Result(embedding=embedding, max_acc_corresponding_metrics=max_acc_corresponding_metrics)
    return result
