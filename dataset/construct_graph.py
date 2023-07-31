# -*- coding: utf-8 -*-
"""
@Time: 2023/5/1 14:24 
@Author: Marigold
@Version: 0.0.0
@Description：
@WeChat Account: Marigold
"""

import numpy as np
from utils.load_data import load_data
from utils.data_processor import construct_graph, numpy_to_torch, torch_to_numpy


def load_adj(dataset_name, k=3):
    feature, label = load_data(root_path="../", dataset_name=dataset_name)
    feature = numpy_to_torch(feature).to("cuda")

    adj = construct_graph(feature, k, metric="heat")
    adj = torch_to_numpy(adj)
    np.save(f"./{dataset_name}/{dataset_name}_{k}_adj.npy", adj, allow_pickle=True)
    return


if __name__ == "__main__":
    load_adj("usps")
