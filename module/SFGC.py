# -*- coding: utf-8 -*-
"""
@Time: 2022/12/2 13:58 
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


class SFGC(Module):
    def __init__(self, input_dim1, input_dim2, input_dim3, input_dim4, input_dim5, output_dim):
        super(SFGC, self).__init__()
        self.mlp1 = Linear(input_dim1 + input_dim2 + input_dim3 + input_dim4 + input_dim5, 5)
        self.scale_GCN = GCN(input_dim1 + input_dim2 + input_dim3 + input_dim4 + input_dim5,
                             output_dim, activeType='no')

    def forward(self, features1, features2, features3, features4, features5, adj):
        # concat
        cat_features = torch.cat((features1, features2, features3, features4, features5), 1)
        # linear
        mlp_features = self.mlp1(cat_features)
        # tanh activate
        activate_features = torch.tanh(mlp_features)
        # softmax
        softmax_features = F.softmax(activate_features, dim=1)
        # normalization
        normalize_features = F.normalize(softmax_features)
        # slice and transpose
        M_i_1 = normalize_features[:, 0].reshape(normalize_features.shape[0], 1)
        M_i_2 = normalize_features[:, 1].reshape(normalize_features.shape[0], 1)
        M_i_3 = normalize_features[:, 2].reshape(normalize_features.shape[0], 1)
        M_i_4 = normalize_features[:, 3].reshape(normalize_features.shape[0], 1)
        M_i_5 = normalize_features[:, 4].reshape(normalize_features.shape[0], 1)
        ones1 = torch.ones(1, features1.shape[1]).cuda()
        ones2 = torch.ones(1, features2.shape[1]).cuda()
        ones3 = torch.ones(1, features3.shape[1]).cuda()
        ones4 = torch.ones(1, features4.shape[1]).cuda()
        ones5 = torch.ones(1, features5.shape[1]).cuda()
        # calculate the weight matrix
        w_1 = torch.mm(M_i_1, ones1)
        w_2 = torch.mm(M_i_2, ones2)
        w_3 = torch.mm(M_i_3, ones3)
        w_4 = torch.mm(M_i_4, ones4)
        w_5 = torch.mm(M_i_5, ones5)
        # concat
        fusion_features = torch.cat((w_1 * features1,
                                     w_2 * features2,
                                     w_3 * features3,
                                     w_4 * features4,
                                     w_5 * features5), 1)
        output = self.scale_GCN(fusion_features, adj)
        return output
