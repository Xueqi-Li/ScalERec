"""
@CreateTime: 2023-05-21 16:53:26
@LastEditTime: 2023-05-21 16:53:28
@Description: 
"""
import torch
import torch.nn.functional as F
from torch_geometric.nn.models import LightGCN
from torch_geometric.nn.conv import LGConv


class RSLightGCN(LightGCN):
    def __init__(self, user_num: int, item_num: int, config):
        super().__init__(user_num + item_num, config["dim"], config["num_layer"])
        self.user_num = user_num
        self.item_num = item_num
        self.lambda_reg = config["decay"]

    def reset_parameters(self):
        torch.nn.init.normal_(self.embedding.weight, std=0.1)  # use original init.
        for conv in self.convs:
            conv.reset_parameters()

    def get_embedding(self, edge_index, edge_weight=None):
        out = super().get_embedding(edge_index)
        out_user, out_item = torch.split(out, [self.user_num, self.item_num])
        return out_user, out_item

    def get_rating_by_users(self, users, edge_index, edge_weight=None):
        out_user, out_item = self.get_embedding(edge_index, edge_weight)
        pred = torch.sigmoid(out_user[users] @ out_item.t())
        return pred

    def get_rating_by_users_sampling(self, users, items, edge_index, edge_weight=None):
        out_user, out_item = self.get_embedding(edge_index, edge_weight)
        out_item = out_item[items]
        pred = torch.sigmoid(out_user[users] @ out_item.t())
        return pred

    def loss_bpr(self, user, pos, neg, edge_index, edge_weight=None):
        """
        - note: loss reg. only requires involved embeddings
        """
        # lightgcn loss
        out_user, out_item = self.get_embedding(edge_index, edge_weight)
        rank_pos = (out_user[user] * out_item[pos]).sum(dim=-1)
        rank_neg = (out_user[user] * out_item[neg]).sum(dim=-1)
        loss_bpr = torch.nn.functional.softplus(rank_neg - rank_pos).mean()

        embedding0 = self.embedding(torch.cat([user, pos + self.user_num, neg + self.user_num]).view(-1))
        loss_reg = embedding0.norm(p=2).pow(2) / float(len(user)) / 2

        return loss_bpr, loss_reg

    def loss_infonce(self, user, pos, neg, edge_index1, edge_weight1, edge_index2, edge_weight2):
        # InfoNCE loss, https://github.com/wujcan/SGL-Torch/blob/main/model/general_recommender/SGL.py
        out_user1, out_item1 = self.get_embedding(edge_index1, edge_weight1)
        out_user2, out_item2 = self.get_embedding(edge_index2, edge_weight2)
        out_user1, out_item1 = F.normalize(out_user1, dim=-1), F.normalize(out_item1, dim=-1)
        out_user2, out_item2 = F.normalize(out_user2, dim=-1), F.normalize(out_item2, dim=-1)

        user = user.unique()
        pos = pos.unique()
        if len(user) > 1024:
            user = user[torch.randperm(len(user))[:1024]]
        if len(pos) > 1024:
            pos = pos[torch.randperm(len(pos))[:1024]]

        emb_user1, emb_user2 = out_user1[user], out_user2[user]
        emb_item1, emb_item2 = out_item1[pos], out_item2[pos]

        pos_score_user = (emb_user1 * emb_user2).sum(dim=-1)  # 1024
        pos_score_item = (emb_item1 * emb_item2).sum(dim=-1)

        # v4
        node2 = edge_index2.view(-1).unique()
        user2 = node2[node2 < self.user_num]
        item2 = node2[node2 >= self.user_num] - self.user_num
        len_perm = min(1024, len(user2), len(item2))
        perm_user = torch.randperm(len(user2))[:len_perm]
        perm_item = torch.randperm(len(item2))[:len_perm]

        user2 = user2[perm_user]
        item2 = item2[perm_item]
        ttl_score_user = torch.matmul(emb_user1, torch.transpose(out_user2[user2], 0, 1))  # (1024, 25855)
        ttl_score_item = torch.matmul(emb_item1, torch.transpose(out_item2[item2], 0, 1))  # (1024, 29308)

        ssl_logits_user = ttl_score_user - pos_score_user[:, None]
        ssl_logits_item = ttl_score_item - pos_score_item[:, None]
        clogits_user = torch.logsumexp(ssl_logits_user / self.ssl_temp, dim=1).sum()
        clogits_item = torch.logsumexp(ssl_logits_item / self.ssl_temp, dim=1).sum()
        infonce_loss = (clogits_user + clogits_item) / float(len(user)) / 2
        return infonce_loss


class RSPPRGo(RSLightGCN):
    def __init__(self, user_num: int, item_num: int, config, **kwargs):
        super().__init__(user_num, item_num, config)
        self.ssl_temp = config["ssl_temp"]
        self.conv = LGConv(**kwargs)

    def get_embedding(self, edge_index, edge_weight=None):
        x = self.embedding.weight
        out = self.conv(x, edge_index, edge_weight)
        out_user, out_item = torch.split(out, [self.user_num, self.item_num])
        return out_user, out_item
