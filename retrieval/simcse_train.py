# -*- coding: utf-8 -*-
# @File  : simcse_train.py
# @Author: zhangxiaoning
# @Date  : 2021/8/25
import logging
import os
import torch
from tqdm import tqdm
from transformers import BertTokenizer, get_linear_schedule_with_warmup, BertConfig
from config import root_path
import pandas as pd
from retrieval.sample import DataLoader
from retrieval.simcse import Model
from retrieval.simcse_data import TrainDataset
from utils import compute_loss
logger = logging.getLogger(__name__)
train_file = os.path.join(root_path, 'data/processed/train.csv')
test_file = os.path.join(root_path, 'data/test.csv')
dev_file = os.path.join(root_path, 'data/processed/dev.csv')

class Trainer:
    def __init__(self):
        self.batch_size = 10
        self.bertmatch_train_epochs = 3
        self.lr = 1e-5
        self.gradient_accumulation_steps = 1
        # 判断是否有可用GPU
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.save_path = os.path.join(root_path, 'result/simcse/unsup_keywords.pth')

        logger.info('using device:{}'.format(self.device))
        # 定义模型超参数
        vocab_file = os.path.join(root_path, 'lib/bert_new/vocab.txt')
        tokenizer = BertTokenizer.from_pretrained(vocab_file)

        logger.info('初始化bertconfig')
        bertconfig = BertConfig.from_pretrained(os.path.join(root_path, 'lib/bert_new/config.json'))
        bertconfig.attention_probs_dropout_prob = 0.1
        bertconfig.hidden_dropout_prob = 0.1

        logger.info('初始化BERT模型')
        model_path = os.path.join(root_path, 'lib/bert_new/pytorch_model.bin')
        self.bert_model = Model(model_path, bertconfig=bertconfig)

        logger.info('定义新的损失函数')
        #允许batch内部的label有一样的，也有不一样的，根据采样获得

        logger.info('将模型发送到计算设备(GPU或CPU)')
        self.bert_model.to(self.device)

        logger.info('加载训练数据')
        self.data = pd.read_csv(train_file, sep='\t')
        # train = TrainDataset(train_file, tokenizer=tokenizer, maxlen=300)
        # print("trainset:%s" %len(train))
        # self.trainloader = DataLoader(train, batch_size=self.batch_size, shuffle=True)
        self.trainloader = DataLoader(self.batch_size, train_file, tokenizer, maxlen=300)
        self.num_samples, self.num_steps = self.trainloader.get_params()

        logger.info(' 声明需要优化的参数')
        self.optim_parameters = list(self.bert_model.parameters())
        self.init_optimizer(lr=self.lr)

    def init_optimizer(self, lr):
        # 用指定的学习率初始化优化器
        self.optimizer = torch.optim.Adam(self.optim_parameters, lr=lr, weight_decay=1e-3)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=self.num_steps + 1000)
            # ((len(self.trainloader) * self.bertmatch_train_epochs) //
            #                     (self.batch_size * self.gradient_accumulation_steps))

    def load_model(self, model, pretrain_model_path):
        checkpoint = torch.load(pretrain_model_path)
        # 模型刚开始训练的时候, 需要载入预训练的BERT

        checkpoint = {k[5:]: v for k, v in checkpoint.items()
                      if k[:4] == "bert" and "pooler" not in k}
        model.load_state_dict(checkpoint, strict=False)
        torch.cuda.empty_cache()
        logger.info("{} loaded!".format(pretrain_model_path))

    def train(self):

        logger.info('starting training')
        self.bert_model.train()
        size = len(self.data)
       
        for batch, data in enumerate(tqdm(self.trainloader)):
            input_ids = data['input_ids'].view(len(data['input_ids']) * 2, -1).to(self.device)
            attention_mask = data['attention_mask'].view(len(data['attention_mask']) * 2, -1).to(self.device)
            token_type_ids = data['token_type_ids'].view(len(data['token_type_ids']) * 2, -1).to(self.device)
            pred = self.bert_model(input_ids, attention_mask, token_type_ids)
            loss = compute_loss(pred, device=self.device)
            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if batch % 10 == 0:
                loss, current = loss.item(), batch * int(len(input_ids) / 2)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        torch.save(self.bert_model.state_dict(), self.save_path)
        logger.info('training finished')


if __name__ == "__main__":
    trainer = Trainer()
    epochs = 4
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        trainer.train()

