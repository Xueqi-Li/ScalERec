"""
@CreateTime: 2023-06-06 10:47:34
@LastEditTime: 2023-06-06 10:47:34
@Description: 
"""
import os
import os.path as osp
import sys
import time
import multiprocessing

from torch.utils.data import Dataset

# from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
import utils
import cppimport

sys.path.append(osp.join(utils.root_path, "sources"))
sampling = cppimport.imp("sampling")
sampling.seed(2023)

import torch


class BaseDataset(Dataset):
    def __init__(self, config):
        super().__init__()
        self.name = config["name"]
        self.path = osp.join(utils.root_path, "data", self.name)
        self.root_path = utils.root_path
        self._init(config)

    def _init(self, config):
        """
        @description: init Dataset parameters
        """
        df_rating = pd.read_csv(osp.join(self.path, "rating.csv"))
        self.user_num = df_rating[config["UID"]].nunique()
        self.item_num = df_rating[config["IID"]].nunique()
        self.train_mask = self._build_mask(len(df_rating), mode="train")
        self.test_mask = self._build_mask(len(df_rating), mode="test")
        self.val_mask = self._build_mask(len(df_rating), mode="val")
        self.users = df_rating[config["UID"]].values
        self.items = df_rating[config["IID"]].values
        self.train_pos = None

    def _build_mask(self, node_num, idx=None, mode="train"):
        """
        @description: build mask for train, val & test
        """
        if idx is None:
            with open(osp.join(self.path, mode + ".txt")) as rfile:
                s = rfile.read()
            idx = np.array([int(i) for i in s.strip().split()])
        mask = np.zeros(node_num, dtype=np.bool_)
        mask[idx] = True
        return mask

    def get_pos_bpr(self):
        users = self.users[self.train_mask]
        items = self.items[self.train_mask]
        train_size = len(users)
        if self.train_pos is None:
            pos = [[] for i in range(self.user_num)]
            for u, i in zip(users, items):
                pos[u].append(i)
            self.train_pos = pos

        return train_size, self.train_pos

    def get_pos_by_users(self, users):
        _, pos = self.get_pos_bpr()
        return [pos[u] for u in users]

    def get_test_dict(self, mode="test"):
        test_dict = {}
        mask = self.test_mask if mode == "test" else self.val_mask
        users = self.users[mask]
        items = self.items[mask]

        for u, i in zip(users, items):
            if u in test_dict:
                test_dict[u].append(i)
            else:
                test_dict[u] = [i]
        return test_dict

    def get_edge_index(self):
        users = self.users[self.train_mask]
        items = self.items[self.train_mask]

        row = np.stack([users, items + self.user_num]).reshape(1, -1)
        col = np.stack([items + self.user_num, users]).reshape(1, -1)
        row = torch.tensor(row).long()
        col = torch.tensor(col).long()
        return torch.cat([row, col])

    def bpr_sampling(self, neg_ratio=1):
        train_size, pos = self.get_pos_bpr()
        sample = sampling.sample_negative(self.user_num, self.item_num, train_size, pos, neg_ratio)
        return sample
