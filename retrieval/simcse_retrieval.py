# -*- coding: utf-8 -*-
# @File  : simcse_retrieval.py
# @Author: zhangxiaoning/home/zj/chatbot/server-test/Smooth_Heart/Generation_Response/retrieval.py
# /home/zj/chatbot/server-test/Smooth_Heart/Generation_Response/SimCSEHnsw
# /home/zj/chatbot/server-test/Smooth_Heart/Generation_Response/match.py
# @Date  : 2021/8/25

import collections
from inspect import trace
import logging
import os
import traceback
import time
import numpy as np
import pandas as pd
from regex import R
import requests
from sklearn.metrics import accuracy_score
import torch
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import BertTokenizer, BertConfig
from config import root_path
import config
from retrieval.hnsw_faiss import HNSW
from retrieval.simcse import Model
from utils import get_all_corpus_sentence
pd.set_option('display.max_rows',500)
pd.set_option('display.max_columns',500)
pd.set_option('display.width',1000)

logger = logging.getLogger(__name__)


class SimCSEHnsw(object):

  def __init__(self, simcse_path, hnsw_model_path=None, corpus_sentences=None, data=None, device=None, k=10):
        self.device = device if device else torch.device("cpu")

        logger.info('初始化SimCSE模型')
        vocab_file = os.path.join(root_path, 'lib/bert_new/vocab.txt')
        self.tokenizer = BertTokenizer.from_pretrained(vocab_file)

        bertconfig = BertConfig.from_pretrained(os.path.join(config.root_path, 'lib/bert_new/config.json'))
        bertconfig.attention_probs_dropout_prob = 0.1
        bertconfig.hidden_dropout_prob = 0.1

        pretrain_model_path = os.path.join(root_path, 'lib/bert_new/pytorch_model.bin')
        self.sentence_model = Model(pretrain_model_path, bertconfig)

        logger.info('加载训练过的Simcse模型权重')
        state_dict = torch.load(simcse_path)
        self.sentence_model.load_state_dict(state_dict)

        self.k = k

        logger.info('获取数据')
        self.corpus_sentences = corpus_sentences

        logger.info('对数据初始化')
        self.corpus_embeddings = self.encode(self.corpus_sentences, batch_size=4)

        logger.info('获取各个中心向量')
        self.center_embeddings, self.center_data, self.label2questionmap = self.get_center(data)

        self.data = data

        self.hnsw = HNSW(corpus_embeddings=self.center_embeddings,
                         ef=config.ef_construction, M=config.M, embedding_dim=bertconfig.hidden_size,
                         model_path=hnsw_model_path,
                         data=self.center_data)

  def search(self, text):
        text_embedding = self.encode(text, batch_size=1)
        text_embedding = text_embedding.astype('float32')
        text_embedding = text_embedding.reshape(-1, 768)
        result = self.hnsw.search(text_embedding, self.k)
        return result

  def encode(self, sentence, batch_size):
        '''默认转成numpy类型的数据进入hnsw'''
        self.sentence_model = self.sentence_model.to(self.device)
        single_sentence = False
        if isinstance(sentence, str):
            sentence = [sentence]
            single_sentence = True

        embedding_list = []
        with torch.no_grad():
            total_batch = len(sentence) // batch_size + (1 if len(sentence) % batch_size > 0 else 0)
            for batch_id in tqdm(range(total_batch)):
                inputs = self.tokenizer(
                    sentence[batch_id * batch_size:(batch_id + 1) * batch_size],
                    padding=True,
                    truncation=True,
                    max_length=300,
                    return_tensors="pt"
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.sentence_model(**inputs)

                embeddings = outputs

                embedding_list.append(embeddings.cpu())
        embeddings = torch.cat(embedding_list, 0)
        embeddings = np.asarray([emb.numpy() for emb in embeddings])

        if single_sentence:
            embeddings = embeddings[0]

        return embeddings

  def get_center(self, data):
    '''获取各个label对应文本编码后的中心向量'''
    train_questions = list(data['fq'].values)
    train_labels = list(data['label'].values)
    label2questionmap = collections.defaultdict(list)
    centers = []
    fqs = []
    labels = []


    for label, question in zip(train_labels, train_questions):
      label2questionmap[label].append(question)
    
    for label in label2questionmap.keys():
      qestions = label2questionmap[label]
      fqs.append(qestions)
      labels.append(label)
      embeddings = self.encode(qestions, batch_size=4)
      center = np.mean(embeddings, axis=0).reshape(1, -1).flatten()
      centers.append(center)
    
    centers = np.asarray(centers)
    
    data = pd.DataFrame({'fq':fqs, 'label':labels})
    return centers, data, label2questionmap

  def evaluate(self, test_file):
    '''计算测试集的准确率'''
    test = pd.read_csv(test_file, sep='\t')
    questions = list(test['fq'].values)
    labels = list(test['label'].values)
    preds = []
    similarquestions = []
    content = []
    wrong = []

    for question, label in zip(questions, labels):
      res = self.search(question)
      label2questions = self.label2questionmap[label]

      index, fqs, pred, _ = res.loc[0,:]

      if pred != label:
        tmp = {}
        tmp['fq'] = question
        tmp['label'] = label
        tmp['label2question'] = label2questions[0]
        tmp['prediction'] = pred
        tmp['similar'] = fqs[0]
      
        wrong.append(tmp)
      preds.append(pred)
      similarquestions.append(fqs[0])
    
    
    count = 0
    for pred, label in zip(preds, labels):
      if pred == label:
        count += 1
    
    content = pd.DataFrame({'fq':questions, 'predition': preds, 'label':labels, 'similar':similarquestions})
    accuracy = count / len(preds)
    return accuracy, content, wrong

  def predict(self, test_file):
    '''
    在剩下的测试集上预测结果
    fq,prediction, similar
    '''
    data = pd.read_csv(test_file, sep='\t')
    questions = list(data['fq'].values)
    predictions = []
    similar = []

    for ques in questions:
      res = self.search(ques)
      index, fqs, pred, _ = res.loc[0,:]
      predictions.append(pred)
      similar.append(fqs[0])
    
    result = pd.DataFrame({'fq':questions, 'prediction':predictions, 'similar':similar})
    return result

  def get_prediction(self, test_file):
    data = pd.read_csv(test_file, sep=',', encoding='GBK')
    ids = list(data['id'].values)
    questions = list(data['fq'].values)
    pred_label = []
    pred_time = []

    for fq in questions:
      start = time.time()
      res = self.search(fq)
      index, fqs, pred, _ = res.loc[0,:]
      offset = time.time() - start
      pred_label.append(pred)
      pred_time.append(offset)
    
    result = pd.DataFrame({'id':ids, 'fq':questions, 'pred_label':pred_label, 'pred_time':pred_time})

    return result



if __name__ == "__main__":
    data_path = os.path.join(root_path, 'data/processed/train.csv')
    test_file = os.path.join(root_path, 'data/processed/test_pre.csv')
    data = pd.read_csv(data_path, sep='\t')
    corpus_sentence = get_all_corpus_sentence(data, True)
    simcse_path = os.path.join(root_path, 'result/ge2e_simcse/unsup_keywords.pth')
    hnsw_path = os.path.join(root_path, 'result/ge2e_simcse/key_center_simcse_hnsw')
    sentence_hnsw = SimCSEHnsw(simcse_path, hnsw_path, corpus_sentence, data, k=10)

    accuracy, content, wrong = sentence_hnsw.evaluate(test_file)
    print(accuracy)
    print(content)
    # print(wrong)

    # raw_test = os.path.join(root_path, 'data/raw/test.csv')
    # to_file = os.path.join(root_path, 'data/raw/test_prediction')
    # result = sentence_hnsw.get_prediction(raw_test)
    # result.to_csv('{}.csv'.format(to_file), index=False, sep=',')

    # left = os.path.join(root_path, 'data/processed/left.csv')

    # to_file = os.path.join(root_path, 'data/processed/left_prediction')
    # result = sentence_hnsw.predict(left)
    # result.to_csv('{}.csv'.format(to_file), index=False, sep='\t')

        

