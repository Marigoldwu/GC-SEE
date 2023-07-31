# -*- coding: utf-8 -*-
"""
@Time: 2023/4/30 16:00 
@Author: Marigold
@Version: 0.0.0
@Description：
@WeChat Account: Marigold
"""
import torch
import torch.nn.functional as F
import numpy as np

from torch.optim import Adam
from sklearn.cluster import KMeans
from model.GCSEE.model import GCSEE
from utils import data_processor
from utils.evaluation import eva
from utils.result import Result
from utils.utils import count_parameters, get_format_variables


def train(args, data, logger):
    # format: [lambda3, lambda4, max_epoch, lr]
    train_params_dict = {"acm": [0.1, 10, 50, 5e-4],
                         "cite": [1000, 1000, 50, 1e-3],
                         "cora": [0.001, 0.001, 50, 6e-4],
                         "dblp": [1, 10, 50, 1e-3],
                         "usps": [10, 10, 200, 3e-3],
                         "amap": [0.001, 0.001, 200, 4e-5]}
    args.lambda3 = train_params_dict[args.dataset_name][0]
    args.lambda4 = train_params_dict[args.dataset_name][1]
    args.max_epoch = train_params_dict[args.dataset_name][2]
    args.lr = train_params_dict[args.dataset_name][3]
    args.embedding_dim = 10

    model = GCSEE(input_dim=args.input_dim,
                  output_dim=args.clusters,
                  embedding_dim=args.embedding_dim,
                  v=1.0).to(args.device)

    logger.info(model)

    pretrain_ae_filename = args.pretrain_ae_save_path + args.dataset_name + ".pkl"
    pretrain_gat_filename = args.pretrain_gat_save_path + args.dataset_name + ".pkl"
    model.ae.load_state_dict(torch.load(pretrain_ae_filename, map_location='cpu'))
    model.gat.load_state_dict(torch.load(pretrain_gat_filename, map_location='cpu'))

    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_epoch)

    M = data.M.to(args.device).float()
    adj_norm = data_processor.normalize_adj(data.adj)
    adj_norm = data_processor.numpy_to_torch(adj_norm).to(args.device).float()
    adj = data_processor.numpy_to_torch(data.adj).to(args.device).float()
    adj_label = adj
    feature = data.feature.to(args.device).float()
    label = data.label

    with torch.no_grad():
        _, _, _, _, h = model.ae(feature)
        _, r = model.gat(feature, adj, M)

    kmeans_r = KMeans(n_clusters=args.clusters, n_init=20)
    kmeans_r.fit_predict(r.data.cpu().numpy())
    model.cluster_layer_r.data = torch.tensor(kmeans_r.cluster_centers_).to(args.device)

    kmeans_h = KMeans(n_clusters=args.clusters, n_init=20)
    kmeans_h.fit_predict(h.data.cpu().numpy())
    model.cluster_layer_h.data = torch.tensor(kmeans_h.cluster_centers_).to(args.device)

    max_acc, embedding, q_h, q_r, q_z = 0, 0, 0, 0, 0
    max_acc_corresponding_metrics = [0, 0, 0, 0]
    for epoch in range(1, args.max_epoch + 1):
        model.train()
        x_bar, A_pred, q_z, q_r, q_h, embedding = model(feature, adj, adj_norm, M)
        ph = data_processor.target_distribution(q_h.data)
        pr = data_processor.target_distribution(q_r.data)

        x_re_loss = F.mse_loss(x_bar, feature)
        a_re_loss = F.binary_cross_entropy(A_pred.view(-1), adj_label.view(-1))
        re_loss = 10 * x_re_loss + a_re_loss

        kl_loss_zr = F.kl_div(q_z.log(), pr, reduction='batchmean')
        kl_loss_rr = F.kl_div(q_r.log(), pr, reduction='batchmean')
        kl_loss1 = kl_loss_zr + kl_loss_rr

        kl_loss_hh = F.kl_div(q_h.log(), ph, reduction='batchmean')
        kl_loss_rh = F.kl_div(q_r.log(), ph, reduction='batchmean')
        kl_loss2 = kl_loss_hh + kl_loss_rh

        loss = args.lambda3 * kl_loss2 + args.lambda4 * kl_loss1 + re_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            model.eval()
            _, _, pred, _, _, embedding = model(feature, adj, adj_norm, M)
            y_pred = pred.data.cpu().numpy().argmax(1)
            acc, nmi, ari, f1 = eva(label, y_pred)
            if acc > max_acc:
                max_acc = acc
                max_acc_corresponding_metrics = [acc, nmi, ari, f1]
            logger.info(get_format_variables(epoch=f"{epoch:0>3d}", acc=f"{acc:0>.4f}", nmi=f"{nmi:0>.4f}",
                                             ari=f"{ari:0>.4f}", f1=f"{f1:0>.4f}"))
    result = Result(embedding=embedding, max_acc_corresponding_metrics=max_acc_corresponding_metrics)
    # Get the network parameters
    logger.info("The total number of parameters is: " + str(count_parameters(model)) + "M(1e6).")
    mem_used = torch.cuda.max_memory_allocated(device=args.device) / 1024 / 1024
    logger.info(f"The max memory allocated to model is: {mem_used:.2f} MB.")
    return result
