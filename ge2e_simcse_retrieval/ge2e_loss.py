# —*- coding: utf-8 -*-
# Author: zsk
# Creator Date: 2021/7/8
# Description: GE2E Loss

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import compute_loss


class GE2ELoss(nn.Module):
    """
    GE2E loss
    """
    def __init__(self):
        # python 2 用法
        super(GE2ELoss, self).__init__()
        # # python 3 用法
        # super().__init__()

        self.ln_w = nn.Parameter(torch.tensor(10.))
        self.ln_b = nn.Parameter(torch.tensor(-5.))

    def forward(self, normed_embedding, n, m):
        """
        前向算法
        :param normed_embedding: 归一化的embedding [N, M, D]
        :param n: speaker num
        :param m: utterance num
        是否可以加入单句表征的对比学习的损失函数
        loss = compute_loss(pred, device=self.device)
        """

        ck = torch.mean(normed_embedding, 1)  # [N, D]
        sums = torch.sum(normed_embedding, 1)      # [N, D]
        e = normed_embedding.view(n * m, -1)       # [N*M, D]

        # 计算与其他中心的相似度
        sims = torch.mm(e, torch.transpose(ck, 0, 1))  # [N*M, N]
        # 与本类中心的相似度计算
        for i in range(n):
            for j in range(m):
                cj = (sums[i] - e[i*m + j]) / (m - 1)  # [D]
                sims[i*m + j][i] = torch.dot(e[i*m + j], cj)  # [1]

        # 相似度矩阵，即标签概率

        # Softmax 版本，适合文本无关
        # sims = torch.sigmoid_(self.ln_w * sims + self.ln_b)  # [N*M, N]

        # # Contrast版本，适合文本相关
        sims = torch.sigmoid_(self.ln_w * sims + self.ln_b)      # [N*M, N]

        # 真实标签
        labels = torch.zeros(size=(n*m,), dtype=torch.int64, device=e.device)
        # 类别标签矩阵
        for i, j in enumerate(range(0, n*m, m)):
            labels[j: j+m] = i

        # 计算GE2E中的contrast版本loss
        loss = torch.zeros(1).to(e.device)
        for i in range(n):
            # Contrast 版本
            sjik = torch.cat((sims[:, :i], sims[:, i + 1:]), dim=-1)
            max_vs, pos = torch.max(sjik, dim=-1)
            for j in range(m):
                max_v = max_vs[i*m + j]
                loss += 1 - sims[i*m + j][i] + max_v

            # Softmax 版本
            # sims_exp = torch.exp(sims)
            # sum_vs = torch.sum(sims_exp, dim=-1)
            # for j in range(m):
            #     sum_v = torch.log(sum_vs[i*m + j])
            #     loss += - sims[i*m + j][i] + sum_v

        # 求均值
        loss /= (m * n)

        # 计算ACC
        _, preds = torch.max(sims, 1)

        num_corrects = torch.sum(preds == labels.data)
        batch_acc = num_corrects / (n * m)

        return loss, batch_acc
