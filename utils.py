"""
@CreateTime: 2023-04-26 10:38:55
@LastEditTime: 2023-04-26 10:38:57
@Description: 
"""
import os.path as osp
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F

import torch_geometric.utils as pygutils

root_path = "./"


def get_time():
    time_now = datetime.now().strftime("%m%d_%H%M")
    return time_now


def minibatch(*tensors, **kwargs):
    # from https://github.com/gusye1234/LightGCN-PyTorch
    batch_size = kwargs.get("batch_size")

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i : i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i : i + batch_size] for x in tensors)


def shuffle(*arrays, **kwargs):
    # from https://github.com/gusye1234/LightGCN-PyTorch
    require_indices = kwargs.get("indices", False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError("All inputs to shuffle must have " "the same length.")

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result


### metrics, based on https://github.com/gusye1234/LightGCN-PyTorch
def recall_precision_at_k(true, label, k):
    """
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
    right = label[:, :k].sum(1)
    rec_n = np.array([len(t) for t in true])
    rec = np.sum(right / rec_n)
    pre = np.sum(right) / k
    return {"recall": rec, "precision": pre}


def mrr_at_k(label, k):
    """
    Mean Reciprocal Rank
    """
    pred_data = label[:, :k]
    scores = np.log2(1.0 / np.arange(1, k + 1))
    pred_data = pred_data / scores
    pred_data = pred_data.sum(1)
    return np.sum(pred_data)


def ndcg_at_k(true, label, k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    assert len(label) == len(true)
    label = label[:, :k]

    label_matrix = np.zeros((len(label), k))
    for i, items in enumerate(true):
        length = k if k <= len(items) else len(items)
        label_matrix[i, :length] = 1
    max_r = label_matrix
    idcg = np.sum(max_r * 1.0 / np.log2(np.arange(2, k + 2)), axis=1)
    dcg = label * (1.0 / np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.0] = 1.0
    ndcg = dcg / idcg
    ndcg[np.isnan(ndcg)] = 0.0
    return np.sum(ndcg)


def get_label(true_list, rec_k_list):
    label_list = []
    for i in range(len(true_list)):
        true = true_list[i]
        rec_k = rec_k_list[i]
        label = list(map(lambda x: x in true, rec_k))
        label = np.array(label).astype("float")
        label_list.append(label)
    return np.array(label_list).astype("float")


def _load_ppr(ppr, data_name, file_k, tau, topk, alpha=0.2):
    """
    @description: load graph based on ppr, top k.
    - by default read 100.txt file and load top k from that
    """
    path_ppr = osp.join(root_path, "ppr", ppr)

    # load topk neighbors and their weights
    alpha = "{:.2f}.txt".format(alpha)
    row, col, values = [], [], []
    with open(osp.join(path_ppr, "_".join([data_name, "ppr_index_" + str(file_k), alpha]))) as file_index:
        with open(osp.join(path_ppr, "_".join([data_name, "ppr_value_" + str(file_k), alpha]))) as file_value:
            for line_index, line_value in zip(file_index.readlines(), file_value.readlines()):
                index = [int(s) for s in line_index.split()]
                value = [float(s) for s in line_value.split()]
                assert index[0] == int(value[0]), "in func load_topppr, unmatch between index and value"
                row += index[:1] * len(index[1:topk])
                col += index[1:topk]
                values += value[1:topk]

    edge_index = torch.tensor([row, col]).long()
    edge_weight = torch.tensor(values).float()

    # filter neighbors whose weight < ppr_tau

    if tau > 0:
        edge_mask = edge_weight > tau
        # print(edge_index.size(), edge_weight.size(), edge_mask.float().sum(), edge_index.max())
        print(
            "in func load_ppr: {}, after filetring with {}, m is {}, in {}".format(
                ppr,
                tau,
                int(edge_mask.float().sum() / (edge_index.max() + 1)),
                data_name,
            )
        )
        edge_index = edge_index[:, edge_mask].long()
        edge_weight = edge_weight[edge_mask].float()
    return edge_index, edge_weight


def _merge(edge_index1, edge_weight1, edge_index2, edge_weight2):
    edge_index = torch.cat([edge_index1, edge_index2], dim=-1)
    edge_weight = torch.cat([edge_weight1, edge_weight2], dim=-1)
    edge_index, edge_weight = pygutils.coalesce(edge_index, edge_weight, reduce="add")
    return edge_index, edge_weight


def load_ppr(ppr, data_name, file_k, tau, topk):
    if ppr in ["topppr", "tarppr"]:
        edge_index, edge_weight = _load_ppr(ppr, data_name, file_k, tau, topk)
    elif ppr == "toptarppr":  # merge top k results
        edge_index_top, edge_weight_top = _load_ppr("topppr", data_name, file_k, tau, topk)
        edge_index_tar, edge_weight_tar = _load_ppr("tarppr", data_name, file_k, tau, topk)
        edge_index, edge_weight = _merge(edge_index_top, edge_weight_top, edge_index_tar, edge_weight_tar)
        assert edge_index.size()[-1] == edge_weight.size()[-1], "coalesce error"

    return edge_index, edge_weight


def load_ppr_weight(ppr, data_name, file_k, tau, topk, num_user, w_u=1, w_i=1):
    if ppr in ["topppr", "tarppr"]:
        edge_index, edge_weight = _load_ppr(ppr, data_name, file_k, tau, topk)
        mask_user = edge_index[0] < num_user
        edge_weight = mask_user * edge_weight * w_u + (~mask_user) * edge_weight * w_i
    elif ppr == "toptarppr":  # merge top k results
        edge_index_top, edge_weight_top = _load_ppr("topppr", data_name, file_k, tau, topk)
        edge_index_tar, edge_weight_tar = _load_ppr("tarppr", data_name, file_k, tau, topk)
        # weight for users and item respectively
        mask_user_top = edge_index_top[0] < num_user
        edge_weight_top = mask_user_top * edge_weight_top * w_u + (~mask_user_top) * edge_weight_top * w_i
        mask_user_tar = edge_index_tar[0] < num_user
        edge_weight_tar = mask_user_tar * edge_weight_tar * w_u + (~mask_user_tar) * edge_weight_tar * w_i
        edge_index, edge_weight = _merge(edge_index_top, edge_weight_top, edge_index_tar, edge_weight_tar)
        assert edge_index.size()[-1] == edge_weight.size()[-1], "coalesce error"

    return edge_index, edge_weight
