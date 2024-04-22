import pickle
import time
from datetime import timedelta
import re
import scipy
import torch
from torch.nn import functional as F
import jieba as jie
import numpy as np

def wam(sentence, w2v_model):
    '''
    @description: 通过word average model 生成句向量
    @param {type}
    sentence: 以空格分割的句子
    w2v_model: word2vec模型
    @return:
    '''
    arr = []
    for s in sentence:
        if s not in w2v_model.wv.vocab.keys():
            arr.append(np.random.randn(200))
        else:
            arr.append(w2v_model.wv.get_vector(s))
    return np.mean(np.array(arr), axis=0).reshape(1, -1)


def load_embedding(cached_embedding_file):
    """load embeddings"""
    with open(cached_embedding_file, mode='rb') as f:
        return pickle.load(f)


def save_embedding(word_embeddings, cached_embedding_file):
    """save word embeddings"""
    with open(cached_embedding_file, mode='wb') as f:
        pickle.dump(word_embeddings, f)

def clean_text(text):
    text = re.sub(u"([hH]ttp[s]{0,1})://[a-zA-Z0-9\.\-]+\.([a-zA-Z]{2,4})(:\d+)?(/[a-zA-Z0-9\-~!@#$%^&*+?:_/=<>.',;]*)?", '', text)  # remove http:xxx
    text = re.sub(u'#[^#]+#', '', text)  # remove #xxx#
    text = re.sub(u'回复@[\u4e00-\u9fa5a-zA-Z0-9_-]{1,30}:', '', text)  # remove "回复@xxx:"
    text = re.sub(u'@[\u4e00-\u9fa5a-zA-Z0-9_-]{1,30}', '', text)  # remove "@xxx"
    text = re.sub(r'[0-9]+', 'DIG', text.strip()).lower()
    text = ''.join(text.split())  # split remove spaces
    return text

def word_tokenize(line):
    content = clean_text(line)
    # content_words = [m for m in jie.lcut(content) if m not in self.stop_words]
    return jie.lcut(content)

def bm25load(fpath):
    f = open(fpath, 'rb')
    bm25Model = pickle.load(f)
    f.close()
    return bm25Model


def bm25save(obj, fpath):
    f = open(fpath, 'wb')
    pickle.dump(obj, f)
    f.close()

def compute_corrcoef(x, y):
    """Spearman相关系数
    """
    return scipy.stats.spearmanr(x, y).correlation


def compute_loss(y_pred, lamda=0.05, device=None):
    idxs = torch.arange(0, y_pred.shape[0], device=device)
    y_true = idxs + 1 - idxs % 2 * 2
    similarities = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=2)
    # torch自带的快速计算相似度矩阵的方法
    similarities = similarities - torch.eye(y_pred.shape[0], device=device) * 1e12
    # 屏蔽对角矩阵即自身相等的loss
    similarities = similarities / lamda
    # 论文中除以 temperature 超参 0.05
    loss = F.cross_entropy(similarities, y_true)
    return torch.mean(loss)


def get_all_corpus_sentence(data, is_retrieval=False):
    '''获取所有的sentence_corpus'''
    # data = pd.read_csv(data_path)
    corpus_sentences = list(data['fq'].values)
    corpus_sentences = list(corpus_sentences)
    if is_retrieval:
        return corpus_sentences[0:100]
    return corpus_sentences