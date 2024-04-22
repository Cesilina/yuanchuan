"""
author:
date:2021/8/6-10:07
"""
from gensim.summarization import bm25
import sys, os
from config import root_path
from utils import bm25load, bm25save, word_tokenize
import pandas as pd
from collections import Counter


class BM25RetrievalModel:
    def __init__(self, corpus, model_path):

        if model_path and os.path.exists(model_path):
            self.model = bm25load(model_path)
        else:
            self.model = bm25.BM25(corpus)
            bm25save(self.model, model_path)

    def get_top_similarities(self, query, topk=10):
        """query: [word1, word2, ..., wordn]"""
        scores = self.model.get_scores(query)
        rtn = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:topk]
        return rtn

    def predict(self, query, k=5):
        """softmax each model predictions and summarize"""
        query = word_tokenize(query)
        res_all = []
        top_index = self.get_top_similarities(query, topk=k)
        return top_index

def evaluate(bm25_model, data_path, labels):
  '''在验证集上计算准确率'''
  data = pd.read_csv(data_path, sep='\t')
  questions = list(data['fq'].values)
  dev_labels = list(data['label'].values)
  results = []
  for question in questions:
        top_index = bm25_model.get_top_similarities(question, topk=5)
        pred = get_label(top_index, labels)
        results.append(pred)

  count = 0
  for pred, label in zip(results, dev_labels):
      if pred == label:
        count += 1
    
  content = pd.DataFrame({'prediction':results, 'label':dev_labels})
  accuracy = count / len(dev_labels)
  return accuracy, content

def get_label(top_index, labels):
    '''寻找出top_index中label出现多数的label值'''
    simi_label = []
    for index, value in top_index:
      simi_label.append(labels[index])
    
    collection_words = Counter(simi_label)
    most_counterNum = collection_words.most_common(3)[0]
    
    pred, value = most_counterNum
    return pred

if __name__ == '__main__':

  train_file = os.path.join(root_path, 'data/train.csv')
  dev_file = os.path.join(root_path, 'data/dev.csv')
  data_path = os.path.join(train_file)
  data = pd.read_csv(data_path, sep=',', encoding='GBK')

  questions_src = list(data['fq'].values)
  questions = [word_tokenize(line) for line in questions_src]

  labels = list(data['label'].values)

  save_model = os.path.join(root_path, 'result/bm25/bm25_retrieval')
  bm25_model = BM25RetrievalModel(questions, save_model)
  # accuracy, content = evaluate(bm25_model, dev_file, labels)
  # print(accuracy)
  # print(content)

  while True:
    query = input(">>:")
    query = word_tokenize(query)
    res_all = []
    top_index = bm25_model.get_top_similarities(query, topk=5)
    #top_index是一个元组(index, bm25数值)
    simi= []
    simi_label = []
    bm25s = []
    for index, value in top_index:
      simi.append(questions_src[index])
      simi_label.append(labels[index])
      bm25s.append(value)
    
    result = pd.DataFrame({'simi_question':simi, 'label': simi_label, 'bm25': bm25s})
    print(result)
  
