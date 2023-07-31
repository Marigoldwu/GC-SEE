# -*- coding: utf-8 -*-
"""
@Time: 2023/4/30 16:00 
@Author: Marigold
@Version: 0.0.0
@Description：
@WeChat Account: Marigold
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parameter import Parameter
from module.AE import AE
from module.FAFGC import FAFGC
from module.GAT_for_GCSEE import GAT
from module.GCN import GCN
from module.SFGC import SFGC


class GCSEE(nn.Module):
    def __init__(self, input_dim, output_dim, embedding_dim, v=1, alpha=0.2):
        super(GCSEE, self).__init__()
        self.ae = AE(input_dim=input_dim,
                     embedding_dim=embedding_dim,
                     enc_1_dim=500,
                     enc_2_dim=500,
                     enc_3_dim=2000,
                     dec_1_dim=2000,
                     dec_2_dim=500,
                     dec_3_dim=500)

        self.gat = GAT(input_dim, embedding_dim, 500, 500, 2000, alpha)

        self.gcn1 = GCN(input_dim, 500)
        self.gcn2 = FAFGC(500, 500)
        self.gcn3 = FAFGC(500, 2000)
        self.gcn4 = FAFGC(2000, 10)

        self.sf = SFGC(500, 500, 2000, 10, 10, output_dim)

        # cluster layer
        self.cluster_layer_r = Parameter(torch.Tensor(output_dim, embedding_dim))
        torch.nn.init.xavier_normal_(self.cluster_layer_r.data)

        # cluster layer
        self.cluster_layer_h = Parameter(torch.Tensor(output_dim, embedding_dim))
        torch.nn.init.xavier_normal_(self.cluster_layer_h.data)

        self.v = v

    def forward(self, x, adj, adj_norm, M):
        A_pred, r = self.gat(x, adj, M)

        q_r = 1.0 / (1.0 + torch.sum(torch.pow(r.unsqueeze(1) - self.cluster_layer_r, 2), 2) / self.v)
        q_r = q_r.pow((self.v + 1.0) / 2.0)
        q_r = (q_r.t() / torch.sum(q_r, 1)).t()

        x_bar, tra1, tra2, tra3, h = self.ae(x)

        q_h = 1.0 / (1.0 + torch.sum(torch.pow(h.unsqueeze(1) - self.cluster_layer_h, 2), 2) / self.v)
        q_h = q_h.pow((self.v + 1.0) / 2.0)
        q_h = (q_h.t() / torch.sum(q_h, 1)).t()

        z1 = self.gcn1(x, adj_norm)  # 500
        z2 = self.gcn2(tra1, z1, adj_norm)  # 500
        z3 = self.gcn3(tra2, z2, adj_norm)  # 2000
        z4 = self.gcn4(tra3, z3, adj_norm)  # 10

        z = self.sf(z1, z2, z3, z4, h, adj_norm)
        predict = F.softmax(z, dim=1)

        return x_bar, A_pred, predict, q_r, q_h, z
