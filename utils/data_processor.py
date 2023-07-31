# -*- coding: utf-8 -*-
# @Author  : Yue Liu
# @Email   : yueliu19990731@163.com
# @Time    : 2022/9/21 0:38
import random

import torch
import numpy as np
import torch.nn.functional as F
from sklearn.preprocessing import normalize


def numpy_to_torch(a, is_sparse=False):
    """
    numpy array to torch tensor

    :param a: the numpy array
    :param is_sparse: is sparse tensor or not
    :return a: torch tensor
    """
    if is_sparse:
        a = torch.sparse.Tensor(a)
    else:
        a = torch.from_numpy(a)
    return a


def torch_to_numpy(t):
    """
    torch tensor to numpy array

    :param t: the torch tensor
    :return t: numpy array
    """
    return t.numpy()


def data_split(full_list, n_sample):
    offset = n_sample
    random.shuffle(full_list)
    len_all = len(full_list)
    index_now = 0
    split_list = []
    while index_now < len_all:
        # 0-2000
        if index_now+offset > len_all:
            split_list.append(full_list[index_now:])
        else:
            split_list.append(full_list[index_now:index_now+offset])
        index_now += offset
    return split_list


def normalize_adj(adj, symmetry=True):
    """
    normalize the adj matrix

    :param adj: input adj matrix
    :param symmetry: symmetry normalize or not
    :return norm_adj: the normalized adj matrix
    """

    # calculate degree matrix and it's inverse matrix
    d = np.diag(adj.sum(0))
    d_inv = np.linalg.inv(d)

    # symmetry normalize: D^{-0.5} A D^{-0.5}
    if symmetry:
        sqrt_d_inv = np.sqrt(d_inv)
        norm_adj = np.matmul(np.matmul(sqrt_d_inv, adj), sqrt_d_inv)

    # non-symmetry normalize: D^{-1} A
    else:
        norm_adj = np.matmul(d_inv, adj)

    return norm_adj


def normalize_adj_torch(adj, symmetry=True):
    """
    Normalize the adjacency matrix.

    :param adj: Input adjacency matrix
    :param symmetry: Symmetry normalize or not
    :return norm_adj: The normalized adjacency matrix
    """
    # Calculate degree matrix and its inverse matrix
    d_inv = torch.diag(1 / torch.sum(adj, dim=1))

    # Symmetry normalize: D^(-0.5) A D^(-0.5)
    if symmetry:
        sqrt_d_inv = torch.sqrt(d_inv)
        norm_adj = torch.matmul(torch.matmul(sqrt_d_inv, adj), sqrt_d_inv)

    # Non-symmetry normalize: D^(-1) A
    else:
        norm_adj = torch.matmul(d_inv, adj)

    return norm_adj


def construct_graph(feat, k=5, metric="euclidean"):
    """
    construct the knn graph for a non-graph dataset

    :param feat: the input feature matrix
    :param k: hyper-parameter of knn
    :param metric: the metric of distance calculation
    - euclidean: euclidean distance
    - cosine: cosine distance
    - heat: heat kernel
    :return knn_graph: the constructed graph
    """

    # euclidean distance, sqrt((x-y)^2)
    if metric == "euclidean" or metric == "heat":
        xy = np.matmul(feat, feat.transpose())
        xx = (feat * feat).sum(1).reshape(-1, 1)
        xx_yy = xx + xx.transpose()
        euclidean_distance = xx_yy - 2 * xy
        euclidean_distance[euclidean_distance < 1e-5] = 0
        distance_matrix = np.sqrt(euclidean_distance)

        # heat kernel, exp^{- euclidean^2/t}
        if metric == "heat":
            distance_matrix = - (distance_matrix ** 2) / 2
            distance_matrix = np.exp(distance_matrix)

    # cosine distance, 1 - cosine similarity
    if metric == "cosine":
        norm_feat = feat / np.sqrt(np.sum(feat ** 2, axis=1)).reshape(-1, 1)
        cosine_distance = 1 - np.matmul(norm_feat, norm_feat.transpose())
        cosine_distance[cosine_distance < 1e-5] = 0
        distance_matrix = cosine_distance

    # top k
    distance_matrix = numpy_to_torch(distance_matrix)
    top_k, index = torch.topk(distance_matrix, k)
    top_k_min = torch.min(top_k, dim=-1).values.unsqueeze(-1).repeat(1, distance_matrix.shape[-1])
    ones = torch.ones_like(distance_matrix)
    zeros = torch.zeros_like(distance_matrix)
    knn_graph = torch.where(torch.ge(distance_matrix, top_k_min), ones, zeros)
    knn_graph = torch_to_numpy(knn_graph)

    return knn_graph


def get_M(adj, t=2):
    """
    calculate the matrix M by the equation:
        M=(B^1 + B^2 + ... + B^t) / t

    :param t: default value is 2
    :param adj: the adjacency matrix
    :return: M
    """
    tran_prob = normalize(adj, norm="l1", axis=0)
    M_numpy = sum([np.linalg.matrix_power(tran_prob, i) for i in range(1, t + 1)]) / t
    return torch.Tensor(M_numpy)


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def pairwise_euclidean_distance(X):
    n, d = X.shape
    X = X.view(n, 1, d)  # 将X的形状调整为(n, 1, d)，以便进行广播计算

    # 计算每对特征向量之间的欧氏距离
    distances = torch.sum((X - X.transpose(0, 1)) ** 2, dim=2)

    return distances


def laplacian_filtering(A, X, t):
    A = A + np.eye(A.shape[0])
    A_norm = normalize_adj(A, symmetry=True)
    identity = torch.eye(A.shape[0])
    Laplacian = identity - A_norm
    for i in range(t):
        X = (identity - Laplacian) @ X
    return X.float()


def comprehensive_similarity(Z1, Z2, E1, E2, alpha):
    Z1_Z2 = torch.cat([torch.cat([Z1 @ Z1.T, Z1 @ Z2.T], dim=1),
                       torch.cat([Z2 @ Z1.T, Z2 @ Z2.T], dim=1)], dim=0)

    E1_E2 = torch.cat([torch.cat([E1 @ E1.T, E1 @ E2.T], dim=1),
                       torch.cat([E2 @ E1.T, E2 @ E2.T], dim=1)], dim=0)

    S = alpha * Z1_Z2 + (1 - alpha) * E1_E2
    return S


def hard_sample_aware_infoNCE(S, M, pos_neg_weight, pos_weight, node_num):
    pos_neg = M * torch.exp(S * pos_neg_weight)
    pos = torch.cat([torch.diag(S, node_num), torch.diag(S, -node_num)], dim=0)
    pos = torch.exp(pos * pos_weight)
    neg = (torch.sum(pos_neg, dim=1) - pos)
    infoNEC = (-torch.log(pos / (pos + neg))).sum() / (2 * node_num)
    return infoNEC


def square_euclid_distance(Z, center):
    ZZ = (Z * Z).sum(-1).reshape(-1, 1).repeat(1, center.shape[0])
    CC = (center * center).sum(-1).reshape(1, -1).repeat(Z.shape[0], 1)
    ZZ_CC = ZZ + CC
    ZC = Z @ center.T
    distance = ZZ_CC - 2 * ZC
    return distance


def high_confidence(Z, center, tao):
    distance_norm = torch.min(F.softmax(square_euclid_distance(Z, center), dim=1), dim=1).values
    value, _ = torch.topk(distance_norm, int(Z.shape[0] * (1 - tao)))
    index = torch.where(distance_norm <= value[-1],
                        torch.ones_like(distance_norm), torch.zeros_like(distance_norm))

    high_conf_index_v1 = torch.nonzero(index).reshape(-1, )
    high_conf_index_v2 = high_conf_index_v1 + Z.shape[0]
    H = torch.cat([high_conf_index_v1, high_conf_index_v2], dim=0)
    H_mat = np.ix_(H.cpu(), H.cpu())
    return H, H_mat


def pseudo_matrix(P, S, node_num, beta, device="cuda"):
    P = torch.tensor(P)
    P = torch.cat([P, P], dim=0)
    Q = (P == P.unsqueeze(1)).float().to(device)
    S_norm = (S - S.min()) / (S.max() - S.min())
    M_mat = torch.abs(Q - S_norm) ** beta
    M = torch.cat([torch.diag(M_mat, node_num), torch.diag(M_mat, -node_num)], dim=0)
    return M, M_mat


def diffusion_adj(adj, mode="ppr", transport_rate=0.2):
    """
    graph diffusion
    :param adj: input adj matrix
    :param mode: the mode of graph diffusion
    :param transport_rate: the transport rate
    - personalized page rank
    -
    :return: the graph diffusion
    """
    # add the self_loop
    adj_tmp = adj + np.eye(adj.shape[0])

    # calculate degree matrix and it's inverse matrix
    d = np.diag(adj_tmp.sum(0))
    d_inv = np.linalg.inv(d)
    sqrt_d_inv = np.sqrt(d_inv)

    # calculate norm adj
    norm_adj = np.matmul(np.matmul(sqrt_d_inv, adj_tmp), sqrt_d_inv)

    # calculate graph diffusion
    if mode == "ppr":
        diff_adj = transport_rate * np.linalg.inv((np.eye(d.shape[0]) - (1 - transport_rate) * norm_adj))
    else:
        diff_adj = None
    return diff_adj


def remove_edge(A, similarity, remove_rate=0.1, device="cuda"):
    """
    remove edge based on embedding similarity
    Args:
        A: the origin adjacency matrix
        similarity: cosine similarity matrix of embedding
        remove_rate: the rate of removing linkage relation
        device:
    Returns:
        Am: edge-masked adjacency matrix
    """
    # remove edges based on cosine similarity of embedding
    n_node = A.shape[0]
    for i in range(n_node):
        A[i, torch.argsort(similarity[i].cpu())[:int(round(remove_rate * n_node))]] = 0

    # normalize adj
    A = A + torch.eye(A.shape[0])
    Am = normalize_adj(A, symmetry=False)
    Am = Am.to(device)
    return Am


def gaussian_noised_feature(X, device="cuda"):
    """
    add gaussian noise to the attribute matrix X
    Args:
        X: the attribute matrix
        device:
    Returns: the noised attribute matrix X_tilde
    """
    N_1 = torch.Tensor(np.random.normal(1, 0.1, X.shape)).to(device)
    N_2 = torch.Tensor(np.random.normal(1, 0.1, X.shape)).to(device)
    X_tilde1 = X * N_1
    X_tilde2 = X * N_2
    return X_tilde1, X_tilde2


def off_diagonal(x):
    """
    off-diagonal elements of x
    Args:
        x: the input matrix
    Returns: the off-diagonal elements of x
    """
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
