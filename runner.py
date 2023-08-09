"""
@CreateTime: 2023-06-07 20:54:52
@LastEditTime: 2023-06-07 20:54:57
@Description: as LightGCN, load all bpr samples to GPU per epoch
"""
import os.path as osp
import logging
import multiprocessing
import time
import random

import torch
from torch import nn, optim

import torch_geometric.utils as pygutils
import numpy as np

import utils
from dataset import BaseDataset

from model import RSLightGCN, RSPPRGo

FLAG_LOG = False
FLAG_OUT = True


class BaseRunner(object):
    """
    @description: runner for full-graph propagation
    """

    def __init__(self, config, **kwargs):
        self.device = config["device"]
        self._init_dataset(config["data"])
        self._init_model(config["model"])
        self._init_logger()

    def _init_dataset(self, config):
        self.path = utils.root_path
        self.dataset = BaseDataset(config)
        self.data_name = self.dataset.name
        self.user_num = self.dataset.user_num
        self.item_num = self.dataset.item_num
        self.edge_index = None
        self.edge_weight = None

    def _init_model(self, config):
        self.model_name = config["name"]
        self.lr = config["lr"]
        self.train_epoch = config["epoch"]
        self.train_batch_size = config["train_batch_size"]
        self.test_batch_size = config["test_batch_size"]
        self.topks = config["topks"]
        self.multicore = config["multicore"]
        self.pretrain = config["pretrain"]
        self.seed = config["seed"]
        self.reg = config["decay"]
        self.epoch_val = config["epoch_val"]
        self.epoch_conv = config["epoch_conv"]
        self.save_checkpoint = config["save_checkpoint"]
        self.load_checkpoint = config["load_checkpoint"]
        self.model = RSLightGCN(self.user_num, self.item_num, config).to(self.device)
        self.opt = optim.Adam(self.model.parameters(), lr=self.lr)

        self.path_log = osp.join(utils.root_path, config["path_log"], "baseline", "_".join(["lg", self.data_name + "_" + str(self.seed)]))
        self._get_full_graph()

    def _init_logger(self):
        """
        @description: ret logger
        """
        self.epoch = 0
        logger = logging.getLogger("logger")
        logger.setLevel(logging.INFO)

        formatter = logging.Formatter("%(asctime)s %(message)s", datefmt="%m-%d %H:%M")
        if FLAG_OUT:
            # stream handler
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        if FLAG_LOG:
            # file handler
            handler = logging.FileHandler(filename=self.path_log + "_" + utils.get_time() + ".log")
            handler.setFormatter(formatter)
            self.file_handler = handler
            logger.addHandler(handler)

        self.logger = logger

    def _get_full_graph(self):
        """
        @description: build full graph for test (small dataset)
        """
        self.edge_index = self.dataset.get_edge_index().to(self.device)

    def _save_checkpoint(self, epoch, best_epoch, best_rec, path_checkpoint=None):
        state = {
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_recall": best_rec,
            "state_dict": self.model.state_dict(),
            "optimizer": self.opt.state_dict(),
        }
        if path_checkpoint is not None:
            torch.save(state, path_checkpoint)
        else:
            torch.save(state, self.path_log + ".pth")

    def _load_checkpoint(self, path_checkpoint=None):
        if path_checkpoint is None:
            path_checkpoint = self.path_log + ".pth"
        checkpoint = torch.load(path_checkpoint, map_location=self.device)
        start_epoch = checkpoint["epoch"]
        best_epoch = checkpoint["best_epoch"]
        best_rec = checkpoint["best_recall"]

        self.model.load_state_dict(checkpoint["state_dict"])  # .to(self.device)
        self.opt.load_state_dict(checkpoint["optimizer"])  # .to(self.device)
        s = "Checkpoint loaded. Resume training from epoch {}".format(start_epoch)
        self.logger.info(s)

        return start_epoch, best_epoch, best_rec

    def _get_loss(self, user, pos, neg, epoch=0):
        """
        @description: calculate loss based on batch
        """
        bpr_loss, reg_loss = self.model.loss_bpr(user, pos, neg, self.edge_index)
        return bpr_loss + self.reg * reg_loss, (bpr_loss, reg_loss)

    def train(self):
        # load model
        start_epoch, best_rec, best_ndcg, best_epoch = 0, 0, 0, 0
        if self.load_checkpoint:
            try:
                start_epoch, best_epoch, best_rec = self._load_checkpoint()
                # start_epoch, best_epoch, best_rec = self._load_checkpoint(
                #     self.epoch_path_load
                # )
            except:
                self.logger.info("No checkpoint file.")
        path_best_model = self.path_log + "_best.model"

        # train model
        # keep random samples same as those without checkpoint
        end_time = time.time() + 24 * 60 * 60
        for epoch in range(start_epoch):
            if time.time() > end_time:
                self.logger.info("STOP at 24h.")
                break
            # sample = self.dataset.bpr_sampling_ppr(epoch=epoch)
            sample = self.dataset.bpr_sampling()
            user = torch.Tensor(sample[:, 0]).long().to(self.device)
            utils.shuffle(user)
        for epoch in range(start_epoch, self.train_epoch):
            # validate model
            self.epoch = epoch
            if epoch % self.epoch_val == 0:
                results = self.test(mode="val")
                self.logger.info(" & ".join(["val", str(epoch), str(results)]))
                rec = results["recall"][0]
                if rec > best_rec:
                    best_rec = rec
                    best_ndcg = results["ndcg"][0]
                    best_epoch = epoch
                    torch.save(self.model.state_dict(), path_best_model)
                else:
                    if epoch - best_epoch > self.epoch_conv or epoch == self.epoch - 1:
                        s = "break at epoch {}, best result is at epoch {}. {:.4f}({:d}) & {:.4f}".format(epoch, best_epoch, best_rec, best_epoch, best_ndcg)
                        self.logger.info(s)
                        break
                if self.save_checkpoint:
                    self._save_checkpoint(epoch, best_epoch, best_rec)

            # bpr sampling
            # sample = self.dataset.bpr_sampling_ppr(epoch=epoch)
            sample = self.dataset.bpr_sampling()
            user = torch.Tensor(sample[:, 0]).long().to(self.device)
            pos = torch.Tensor(sample[:, 1]).long().to(self.device)
            neg = torch.Tensor(sample[:, 2]).long().to(self.device)
            user, pos, neg = utils.shuffle(user, pos, neg)
            total_batch = len(user) // self.train_batch_size + 1
            # mini-batch training
            aver_loss = None
            self.model.train()
            for batch_i, (batch_user, batch_pos, batch_neg) in enumerate(utils.minibatch(user, pos, neg, batch_size=self.train_batch_size)):
                loss, loss_log = self._get_loss(batch_user, batch_pos, batch_neg, epoch)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                if aver_loss is None:
                    aver_loss = np.zeros(len(loss_log))
                for i, l in enumerate(loss_log):
                    aver_loss[i] += l

            aver_loss /= total_batch
            s = " & ".join(["loss"] + ["{:.4f}".format(l) for l in aver_loss])
            # print(s)
            # self.logger.info(s)

        # test
        self.model.load_state_dict(torch.load(path_best_model))
        results = self.test()
        s = "test & {:.4f}({:d}) & {:.4f}".format(results["recall"][0], best_epoch, results["ndcg"][0])
        self.logger.info(s)

    def _get_batch(self, nodes, edge_index, edge_weight=None):
        mask_node = pygutils.index_to_mask(nodes, size=self.dataset.user_num + self.dataset.item_num)
        mask_edge = mask_node[edge_index[0]]
        if edge_weight is not None:
            return edge_index[:, mask_edge], edge_weight[mask_edge]
        else:
            return edge_index[:, mask_edge], None

    def _get_K_rec_sampling(self, batch_users, max_K):
        """
        @description: NOT filter positive items
        """
        sampled_items = torch.randperm(self.item_num)[: self.item_num // 100]
        batch_users_gpu = torch.Tensor(batch_users).long().to(self.device)
        sampled_items = sampled_items.to(self.device)
        nodes = torch.cat([batch_users_gpu, sampled_items + self.dataset.user_num]).unique()
        batch_index, batch_weight = self._get_batch(nodes, self.edge_index, self.edge_weight)
        rating = self.model.get_rating_by_users_sampling(batch_users_gpu, sampled_items, batch_index, batch_weight)

        _, rec_K = torch.topk(rating, k=max_K)  # item index
        rating = rating.cpu().numpy()
        del rating, batch_users_gpu
        return sampled_items[rec_K]

    def _get_K_rec(self, batch_users, max_K):
        allPos = self.dataset.get_pos_by_users(batch_users)
        batch_users_gpu = torch.Tensor(batch_users).long().to(self.device)
        rating = self.model.get_rating_by_users(batch_users_gpu, self.edge_index, self.edge_weight)

        exclude_index = []  # 3307M, 4429 (2055M)
        exclude_items = []
        for range_i, items in enumerate(allPos):
            exclude_index.extend([range_i] * len(items))
            exclude_items.extend(items)
        rating[exclude_index, exclude_items] = -(1 << 10)
        _, rec_K = torch.topk(rating, k=max_K)  # item index
        rating = rating.cpu().numpy()
        del rating, batch_users_gpu
        return rec_K

    def test(self, mode="test"):
        self.model.eval()

        test_dict = self.dataset.get_test_dict(mode)
        max_K = max(self.topks)
        results = {
            "precision": np.zeros(len(self.topks)),
            "recall": np.zeros(len(self.topks)),
            "ndcg": np.zeros(len(self.topks)),
        }

        with torch.no_grad():  # 785M (717M)
            users = list(test_dict.keys())
            try:
                assert self.test_batch_size <= len(users) / 10
            except AssertionError:
                print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
            user_list = []
            rec_list = []
            true_list = []
            total_batch = len(users) // self.test_batch_size + 1
            # data preprocessing
            for batch_users in utils.minibatch(users, batch_size=self.test_batch_size):
                true = [test_dict[u] for u in batch_users]
                if self.data_name in ["gowalla", "yelp2018"]:
                    rec_K = self._get_K_rec(batch_users, max_K)
                else:
                    rec_K = self._get_K_rec_sampling(batch_users, max_K)
                user_list.append(batch_users)
                true_list.append(true)
                rec_list.append(rec_K.cpu())

            # evaluation
            assert total_batch == len(user_list)
            if self.multicore == 1:  # multiprocessing
                CORES = int(multiprocessing.cpu_count() / 1.2)
                rets = []
                pool = multiprocessing.Pool(CORES)
                for rec, true in zip(rec_list, true_list):
                    rets.append(
                        pool.apply_async(
                            self.test_one_batch,
                            (
                                rec,
                                true,
                                self.topks,
                            ),
                        )
                    )
                pool.close()
                pool.join()

                for ret in rets:
                    result = ret.get()
                    results["recall"] += result["recall"]
                    results["precision"] += result["precision"]
                    results["ndcg"] += result["ndcg"]
            else:
                pre_results = []
                for rec_k, true in zip(rec_list, true_list):
                    pre_results.append(self.test_one_batch(rec_k, true, self.topks))
                for result in pre_results:
                    results["recall"] += result["recall"]
                    results["precision"] += result["precision"]
                    results["ndcg"] += result["ndcg"]
            results["recall"] = np.around(results["recall"] / float(len(users)), 4).tolist()
            results["precision"] = np.around(results["precision"] / float(len(users)), 4).tolist()
            results["ndcg"] = np.around(results["ndcg"] / float(len(users)), 4).tolist()

            return results

    @staticmethod
    def test_one_batch(rec_k, true, topks):
        """
        - test the model on batched users
        """
        label = utils.get_label(true, rec_k)
        pre, recall, ndcg = [], [], []

        for k in topks:
            ret = utils.recall_precision_at_k(true, label, k)
            pre.append(ret["precision"])
            recall.append(ret["recall"])
            ndcg.append(utils.ndcg_at_k(true, label, k))

        return {
            "recall": np.array(recall),
            "precision": np.array(pre),
            "ndcg": np.array(ndcg),
        }

    def get_epoch_time(self, skip=2, active=10):
        start = 0
        for i in range(skip + active):
            epoch_start = time.time()
            if i == skip:
                start = time.time()
            sample = self.dataset.bpr_sampling()
            # sample = self.dataset.bpr_sampling_ppr()
            user = torch.Tensor(sample[:, 0]).long().to(self.device)
            pos = torch.Tensor(sample[:, 1]).long().to(self.device)
            neg = torch.Tensor(sample[:, 2]).long().to(self.device)
            user, pos, neg = utils.shuffle(user, pos, neg)

            self.model.train()
            for batch_i, (batch_user, batch_pos, batch_neg) in enumerate(utils.minibatch(user, pos, neg, batch_size=self.train_batch_size)):
                loss, _ = self._get_loss(batch_user, batch_pos, batch_neg)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                # if epoch_time>30 min, stop
                if time.time() - epoch_start > 30 * 60:
                    return 0
            print("epoch {} done. {:.4f}".format(i, time.time() - start))
            # t += time.time() - start
        t = time.time() - start
        return t / active


class PPRRunner(BaseRunner):
    """
    @description: PPR-based subgraph propagation
    """

    def __init__(self, config):
        super().__init__(config)
        # ppr parameter

    def _init_model(self, config):
        # ppr related
        self.ppr = config["ppr"]
        self.ppr_k = config["ppr_k"]
        self.ppr_alpha = config["ppr_alpha"]
        self.ppr_tau = config["ppr_tau"]
        self.ppr_norm = config["ppr_norm"]
        self.file_k = config["ppr_file_k"]

        # basic model
        self.model_name = config["name"]
        self.lr = config["lr"]
        self.train_epoch = config["epoch"]
        self.train_batch_size = config["train_batch_size"]
        self.test_batch_size = config["test_batch_size"]
        self.topks = config["topks"]
        self.multicore = config["multicore"]
        self.pretrain = config["pretrain"]
        self.seed = config["seed"]
        self.reg = config["decay"]
        self.epoch_val = config["epoch_val"]
        self.epoch_conv = config["epoch_conv"]
        self.save_checkpoint = config["save_checkpoint"]
        self.load_checkpoint = config["load_checkpoint"]
        self.model = RSPPRGo(self.user_num, self.item_num, config).to(self.device)
        self.opt = optim.Adam(self.model.parameters(), lr=self.lr)
        self.path_log = osp.join(
            utils.root_path,
            config["path_log"],
            "_".join([self.data_name, self.ppr, str(self.seed)]),
        )
        self._get_full_graph()

    def _get_loss(self, user, pos, neg, epoch=0):
        nodes = torch.cat([user, pos + self.dataset.user_num, neg + self.dataset.user_num]).unique()
        batch_index, batch_weight = self._get_batch(nodes, self.edge_index, self.edge_weight)
        bpr_loss, reg_loss = self.model.loss_bpr(user, pos, neg, batch_index, batch_weight)
        loss = bpr_loss + self.reg * reg_loss
        return loss, (bpr_loss, reg_loss)

    def _get_full_graph(self):
        """
        @description: build full graph for test (small dataset)
        """
        self.edge_index, self.edge_weight = utils.load_ppr(self.ppr, self.data_name, self.file_k, self.ppr_tau, self.ppr_k)

        # self.edge_index = self.edge_index[[1, 0]]

        self.edge_index = self.edge_index.to(self.device)
        self.edge_weight = self.edge_weight.to(self.device)


class SSLRunner(PPRRunner):
    """
    @description: PPR-based subgraph propagation + Contrastive Learning
    """

    def __init__(self, config):
        super().__init__(config)

    def _init_model(self, config):
        # ppr related
        self.ppr = config["ppr"]
        self.ppr_k = config["ppr_k"]
        self.ppr_alpha = config["ppr_alpha"]
        self.ppr_tau = config["ppr_tau"]
        self.ppr_norm = config["ppr_norm"]
        self.ssl_reg = config["ssl_reg"]
        self.file_k = config["ppr_file_k"]

        self.bippr_u = 1
        self.bippr_i = 1
        # basic model
        self.model_name = config["name"]
        self.lr = config["lr"]
        self.train_epoch = config["epoch"]
        self.train_batch_size = config["train_batch_size"]
        self.test_batch_size = config["test_batch_size"]
        self.topks = config["topks"]
        self.multicore = config["multicore"]
        self.pretrain = config["pretrain"]
        self.seed = config["seed"]
        self.reg = config["decay"]
        self.epoch_val = config["epoch_val"]
        self.epoch_conv = config["epoch_conv"]
        self.save_checkpoint = config["save_checkpoint"]
        self.load_checkpoint = config["load_checkpoint"]
        self.model = RSPPRGo(self.user_num, self.item_num, config).to(self.device)
        self.opt = optim.Adam(self.model.parameters(), lr=self.lr)
        # overall
        self.path_log = osp.join(
            utils.root_path,
            config["path_log"],
            "_".join([self.data_name, "ssl"]),
        )

        # para: ppr_k
        # self.path_log = osp.join(utils.root_path, config["path_log"], "_".join([self.data_name, str(self.ppr_k)]))
        # # para: ppr_tau
        # self.path_log = osp.join(
        #     utils.root_path, config["path_log"], "_".join([self.data_name, str(self.ppr_k), str(self.ppr_tau)])
        # )
        # # para: reg_ssl
        # self.path_log = osp.join(
        #     utils.root_path, config["path_log"], "_".join([self.data_name, str(self.ssl_reg)])
        # )
        # # para: reg_l2
        # self.path_log = osp.join(utils.root_path, config["path_log"], "_".join([self.data_name, str(self.reg)]))

        self._get_full_graph()

    def _get_loss(self, user, pos, neg, epoch=10):
        """
        @description: generate mini-batch edge & cal loss. (based on mini-batch user, pos & neg)
        """
        # get batch data
        nodes = torch.cat([user, pos + self.dataset.user_num, neg + self.dataset.user_num]).unique()
        batch_top_index, batch_top_weight = self._get_batch(nodes, self.edge_index[:, self.top_mask], self.edge_weight[self.top_mask])
        batch_tar_index, batch_tar_weight = self._get_batch(nodes, self.edge_index[:, ~self.top_mask], self.edge_weight[~self.top_mask])
        batch_index, batch_weight = utils._merge(batch_top_index, batch_top_weight, batch_tar_index, batch_tar_weight)
        # cal loss
        bpr_loss, reg_loss = self.model.loss_bpr(user, pos, neg, batch_index, batch_weight)
        dict_data_span = {"gowalla": 20, "yelp2018": 20, "book": 20, "taobao": 100}
        span_ssl = dict_data_span[self.data_name]
        span_ssl = 20
        if self.ssl_reg > 0 and random.randint(0, span_ssl - 1) % span_ssl == 0:
            infonce_loss = self.model.loss_infonce(
                user,
                pos,
                neg,
                batch_top_index,
                batch_top_weight,
                batch_tar_index,
                batch_tar_weight,
            )
        else:
            infonce_loss = 0

        loss = bpr_loss + self.reg * reg_loss + self.ssl_reg * infonce_loss
        return loss, (bpr_loss, reg_loss, infonce_loss)

    def _get_full_graph(self):
        """
        @description: build full graph for test (small dataset)
        """
        top_index, top_weight = utils.load_ppr("topppr", self.data_name, self.file_k, self.ppr_tau, self.ppr_k)
        tar_index, tar_weight = utils.load_ppr_weight("tarppr", self.data_name, self.file_k, self.ppr_tau, self.ppr_k, self.user_num, self.bippr_u, self.bippr_i)

        edge_index = torch.cat([top_index, tar_index], dim=-1)
        edge_weight = torch.cat([top_weight, tar_weight], dim=-1)
        self.edge_index = edge_index.long().to(self.device)
        self.edge_weight = edge_weight.float().to(self.device)

        top_mask = torch.tensor([1] * len(top_weight) + [0] * len(tar_weight))
        self.top_mask = top_mask.bool().to(self.device)
