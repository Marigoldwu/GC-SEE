# -*- coding: utf-8 -*-
"""
@Time: 2022/12/2 13:05 
@Author: Marigold
@Version: 0.0.0
@Description：
@WeChat Account: Marigold
"""

import torch
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn import Linear

from module.GCN import GCN


class FAFGC(Module):
    def __init__(self, input_dim, output_dim):
        """
        :param input_dim: the dimension of input features
        :param output_dim: the dimension of output features
        """
        super(FAFGC, self).__init__()
        self.mlp = Linear(input_dim * 2, 2)
        self.gcn = GCN(input_dim, output_dim)

    def forward(self, input_features1, input_features2, adj):
        """
        :param input_features1: input features 1
        :param input_features2: input features 2
        :param adj: the Symmetric normalized Laplace matrix
        :return: gcn_output_features
        """
        # concat
        cat_features = torch.cat((input_features1, input_features2), 1)
        mlp_features = self.mlp(cat_features)
        activate_features = torch.tanh(mlp_features)
        softmax_features = F.softmax(activate_features, dim=1)
        normalize_features = F.normalize(softmax_features)

        # slice and transpose

        M_i_1 = normalize_features[:, 0].reshape(normalize_features.shape[0], 1)
        M_i_2 = normalize_features[:, 1].reshape(normalize_features.shape[0], 1)
        ones = torch.ones(1, input_features1.shape[1]).cuda()

        # calculate the weight matrix
        w_1 = torch.mm(M_i_1, ones)
        w_2 = torch.mm(M_i_2, ones)

        # fuse features
        fusion_features = w_1 * input_features1 + w_2 * input_features2

        # gcn
        gcn_output_features = self.gcn(fusion_features, adj)
        return gcn_output_features
