# -*- coding: utf-8 -*-
# @File  : data_utils.py
# @Author: zhangxiaoning
# @Date  : 2021/8/25

import collections
from torch.utils.data import Dataset
import pandas as pd
from pyhanlp import HanLP


class TrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, maxlen):
        self.corpus, data = self.get_all_corpus_sentence(data_path)  # get_all_corpus_sentence(data_path)(-->闲聊数据集)
        self.data = data
        self.tokenizer = tokenizer
        self.maxlen = maxlen

    def text_to_id(self, source):
        sample = self.tokenizer([source, source], max_length=self.maxlen, truncation=True, padding='max_length',
                                return_tensors='pt')
        return sample

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.text_to_id(self.data[idx])

    def get_all_corpus_sentence(self, data_path, is_retrieval=False):
        '''
        获取所有的sentence_corpus
        无监督学习，排除有相同label的数据
        '''
        data = pd.read_csv(data_path, sep='\t')
        corpus = list(data['fq'].values)
        corpus = list(corpus)
        content = []
        label_list = []
        if is_retrieval:
            return corpus[0:100]
        for row in data.iterrows():
            fq = row[1]['fq']
            label = row[1]['label']
            if label not in label_list:
              label_list.append(label)
              content.append(fq)
        return corpus, content
    
    # def combine(self, ):
