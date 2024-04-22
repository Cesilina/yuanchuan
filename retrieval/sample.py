# —*- coding: utf-8 -*-
# Author: zsk
# Creator Date: 2021/7/8
# Description: 用于GE2E数据加载


import collections
import torch
import random
import os
import numpy as np
import pandas as pd


class DataSampler(object):
    """
    用户数据采样
    n表示类别数
    m表示在类别中采样的样本数
    """

    def __init__(self, n, m, datapath):
        self.n = n
        self.m = m
        self.label2questionmap = self.load_file(datapath)
        self.labels = self.label2questionmap.keys()

        self.total_samples_num = 0
        for label in self.labels:
            self.total_samples_num += len(self.label2questionmap[label])

    def sample(self):
        """
        数据采样
        """
        data = []
        labels = random.sample(self.labels, self.n)

        for label in labels:
            fq = random.sample(self.label2questionmap[label], self.m)
            data.append(fq)
    
        return data

    def load_file(self, file_path):
        """
        读取数据集
        获取label2questions的映射
        """
        label2questionmap = collections.defaultdict(list)
        train = pd.read_csv(file_path, sep='\t')
        train_labels = list(train['label'].values)
        train_questions = list(train['fq'].values)

        for label, question in zip(train_labels, train_questions):
          label2questionmap[label].append(question)

        return label2questionmap




class DataLoader(object):
    """
    数据加载器
    """
    def __init__(self, batch_size, datadir, tokenizer, maxlen):
        self.n = batch_size
        self.m = 1
        self.datadir = datadir
        self.sampler = DataSampler(self.n, self.m, self.datadir)
        self.tokenizer = tokenizer
        self.maxlen = maxlen

        self.num_samples = self.sampler.total_samples_num
        self.num_steps = self.num_samples / (self.m * self.n)

    def __iter__(self):
        for _ in range(int(self.num_steps)):
            yield self.encode(self.sampler.sample())

    def encode(self, samples):
        '''
        input_ids
        attention_mask
        token_type_ids
        '''
        input_ids = []
        attention_mask = []
        token_type_ids = []
        results = {}
        for sample in samples:
            sample = ''.join(sample)
            result = self.text_to_id(sample)

            input_ids.append(result['input_ids'].numpy().flatten())
            attention_mask.append(result['attention_mask'].numpy().flatten())
            token_type_ids.append(result['token_type_ids'].numpy().flatten())
        
        results['input_ids'] = torch.tensor(input_ids)
        results['attention_mask'] = torch.tensor(attention_mask)
        results['token_type_ids'] = torch.tensor(token_type_ids)
        return results

    def text_to_id(self, source):
        sample = self.tokenizer([source, source], max_length=self.maxlen, truncation=True, padding='max_length',
                                return_tensors='pt')
        return sample

    def get_params(self):
        """
        获取dataloader参数
        :return: num_epoch, num_samples, num_steps
        """
        return self.num_samples, self.num_steps
