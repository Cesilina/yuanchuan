# -*- coding: utf-8 -*-
# @File  : simcse.py
# @Author: zhangxiaoning
# @Date  : 2021/8/25
import torch
from torch import nn
from tqdm import tqdm
from transformers import BertModel, BertConfig
import logging

import config

logger = logging.getLogger(__name__)


class Model(nn.Module):
    def __init__(self, model_path, bertconfig: BertConfig):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(model_path, config=bertconfig)

    def forward(self, input_ids, attention_mask, token_type_ids):
        x1 = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        output = x1.pooler_output

        if config.output_way == 'cls':
            output = x1.last_hidden_state[:, 0]

        return output
