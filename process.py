import collections
from lib2to3.pgen2 import token
from lib2to3.pgen2.tokenize import tokenize
import math
from operator import index

from pyparsing import col
from transformers import BertConfig, BertTokenizer
from config import root_path
import pandas as pd
import os
from random import sample
import matplotlib.pyplot as plt

import jieba.posseg as pseg
from pyhanlp import HanLP, JClass
from retrieval.simcse import Model
from gensim.models import KeyedVectors
import json
from sklearn.metrics.pairwise import cosine_similarity


from text_normalize import TextNormalizer
from utils import wam
data_file = os.path.join(root_path, 'data/raw/train.csv')
# train_to_file = os.path.join(root_path, 'data/processed/train.csv')
# test_file = os.path.join(root_path, 'data/raw/test.csv')
# dev_to_file = os.path.join(root_path, 'data/processed/dev.csv')

# test_to_file = os.path.join(root_path, 'data/processed/test_pre')
# questions = []
# labels = []

# for i in range(1, 11):
#   path = os.path.join(root_path, 'data/samples/{s}.csv'.format(s=i))
 
#   data = pd.read_csv(path, sep=',', encoding='GBK')
#   fqs = list(data['fq'].values)
#   pred_labels = list(data['pred_label'].values)

#   for fq, label in zip(fqs, pred_labels):
#     print(label)
#     if math.isnan(label):
#       continue
#     questions.append(fq)
#     labels.append(int(label))

# data = pd.DataFrame({'fq':questions, 'label':labels})
# data.to_csv('{}.csv'.format(test_to_file), index=False, sep='\t')


# data = '介绍一下外汇电子账户'
# words = pseg.cut(data)
# for word, flag in words:
#   print(word)
#   print(flag)

# pylp = HanLP.segment(data)
# print(pylp)

# py_words = HanLP.extractKeyword(data, 5)
# print(py_words)

# CoreSynonymDictionary = JClass("com.hankcs.hanlp.dictionary.CoreSynonymDictionary")

# print(CoreSynonymDictionary.rewrite(data), '\n')
# print("获取同义词",CoreSynonymDictionary.rewrite("收不到短信验证码微信无法绑定银行卡"))




#读取训练集，抽取关键词，加入bert中的vocab.txt
# vocab_file = os.path.join(root_path, 'lib/bert/vocab.txt')
# vocab = []
# with open(vocab_file, 'r') as f:
#   lines = f.readlines()
#   for line in lines:
#     line = line.strip().split('\t')
#     line = ''.join(line)
#     vocab.append(line)

# def is_invocab(word):
#   return word not in vocab


# print(len(vocab))
# data = pd.read_csv(train_to_file, sep='\t')
# questions = list(data['fq'].values)
# keywords = []
# for ques in questions:
#   pywords = HanLP.extractKeyword(ques, 3)
#   # print(pywords)
#   pywords = filter(is_invocab, pywords)
#   keywords.extend(pywords)
# # print(keywords)

# #将keywords加入vocab.txt中
# # vocab.extend(keywords)
# # print(len(vocab))
# keywords = ['活期', 'atm', '自助设备']

# bertconfig = BertConfig.from_pretrained(os.path.join(root_path, 'lib/bert_new/config.json'))
# bertconfig.attention_probs_dropout_prob = 0.1
# bertconfig.hidden_dropout_prob = 0.1

# vocab_file = os.path.join(root_path, 'lib/bert_new/vocab.txt')
# tokenizer = BertTokenizer.from_pretrained(vocab_file)
# model_path = os.path.join(root_path, 'lib/bert_new/pytorch_model.bin')
# bert_model = Model(model_path, bertconfig=bertconfig)

# num_added_toks = tokenizer.add_tokens(keywords)
# bert_model.bert.resize_token_embeddings(len(tokenizer))
# tokenizer.save_pretrained(os.path.join(root_path, 'lib/bert_new/'))

# data = '请问如何将活期存款转存为定期的'
# print(tokenizer.tokenize(data))


# to_file = os.path.join(root_path, 'lib/bert_new/vocab.txt')
# with open(to_file, 'w') as f:
#   for word in vocab:
#     f.write(word)
#     f.write('\n')


#将data的最后1000条数据作为验证集
# print(data[-1000:])
# data[-1000:].to_csv('{}.csv'.format(to_file), index=False, sep='\t')
# train_questions = list(data['fq'].values)
# train_labels = list(data['label'].values)

data = pd.read_csv(data_file, sep=',', encoding='GBK')
stats = data['label'].value_counts()
print(data['label'].max())
print(stats)
# stats = stats.sort_index()
# x = list(stats.index)
# y = list(stats.values)
# plt.plot(x, y)
# plt.show()
# dev = []
# train = []

# #在train中抽取验证集，至少保证每一类有一个
# labelmap = dict(zip(x, y))
# label2questionmap = collections.defaultdict(list)

# for label, question in zip(train_labels, train_questions):
#   label2questionmap[label].append(question)
# print(label2questionmap)

# dev_questions = []
# for label in label2questionmap.keys():
#   questions = label2questionmap[label]
#   #随机抽取1条加入验证集
#   ques = sample(questions, 1)
#   tmp = {}
#   tmp['label'] = label
#   tmp['fq'] = ques[0]
#   dev.append(tmp)
#   dev_questions.append(ques[0])


# for label, question in zip(train_labels, train_questions):
#   if question in dev_questions:
#     continue
#   tmp = {}
#   tmp['label'] = label
#   tmp['fq'] = question
#   train.append(tmp)

# train = pd.DataFrame(train, columns=['fq', 'label'])
# dev = pd.DataFrame(dev, columns=['fq', 'label'])



# train = pd.read_csv(train_to_file, sep='\t')




#对训练集中的繁体字改为简体字
# tool = TextNormalizer()

# train = pd.read_csv(dev_to_file, sep='\t')
# questions = list(train['fq'].values)
# labels = list(train['label'].values)

# new_ques = []

# for ques in questions:
#   quest = tool.to_simple(ques)
#   new_ques.append(quest)

# questions = [tool.to_simple(question) for question in questions]
# data = pd.DataFrame({'fq':new_ques, 'label':labels}, columns=['fq', 'label'])

# to_file = os.path.join(root_path, 'data/processed/dev_pre')
# data.to_csv('{}.csv'.format(to_file), index=False, sep='\t')
# dev.to_csv('{}.csv'.format(dev_to_file), index=False, sep='\t')

#讲训练集和验证集合并在一起训练
# to_file= os.path.join(root_path, 'data/train/train')
# train = pd.read_csv(train_to_file, sep='\t')
# test = pd.read_csv(test_file, sep=',', encoding='GBK')
# train_fq = list(train['fq'].values)
# test_fq = list(test['fq'].values)
# all_fq = []
# all_fq.extend(train_fq)
# all_fq.extend(test_fq)
# print(len(all_fq))
# all_fq = pd.DataFrame(all_fq, columns=['fq'])
# all_fq.to_csv('{}.csv'.format(to_file), index=False, sep='\t')


#拆分测试集，打标，共1566条样本
# test_result = os.path.join(root_path, 'data/processed/test_result_compare.json')
# test_file = os.path.join(root_path, 'data/raw/test.csv')
# test = pd.read_csv(test_file, sep=',', encoding='GBK')
# questions = list(test['fq'].values)

# with open(test_result, 'r') as f:
#   data = json.load(f)
#   # print(data)
#   content = data['data']
#   #对content进行分组，写入10个文件中
#   for i in range(0, len(content), 156):
#     if i == 1404:
#       sample = content[i:]
#       sample_test = questions[i:]
#     else:
#       sample = content[i:i + 156]
#       sample_test = questions[i:i+156]

#     path = os.path.join(root_path, 'data/samples/{s}.json'.format(s=i))
#     to_file = os.path.join(root_path, 'data/samples/{s}'.format(s=i))

#     sample_test = pd.DataFrame(sample_test, columns=['fq'])
#     sample_test.to_csv('{}.csv'.format(to_file), index=False, sep='\t')


#     json_str = json.dumps(sample, indent = 4, ensure_ascii=False)
#     with open(path, 'w') as json_file:
#         json_file.write(json_str)
# w2v_path = os.path.join(root_path, 'model/word2vec/Word_Embedding/Tencent_AILab_ChineseEmbedding_1000000.bin')
# w2v_model = KeyedVectors.load(w2v_path)

# text1 = ['账户', '外汇', '开']
# text2 = ['销户', '外汇', '办理']
# text3 = ['电子', '介绍', '外汇', '账户']
# sent1 = wam(text1, w2v_model)
# sent2 = wam(text2, w2v_model)
# sent3 = wam(text3, w2v_model)
# print(sent1.shape)
# print(sent2.shape)

# similarities = cosine_similarity(sent1, sent2)
# print(similarities)

# similarities = cosine_similarity(sent1, sent3)
# print(similarities)


# similar = w2v_model.similarity(word1, Word2)
# print(similar)
# train_file = os.path.join(root_path, 'data/processed/train.csv')
# dev_file = os.path.join(root_path, 'data/processed/dev.csv')
# train = pd.read_csv(train_file, sep='\t')
# dev = pd.read_csv(dev_file, sep='\t')
# questions = list(train['fq'].values)
# labels = list(train['label'].values)

# fqs = list(dev['fq'].values)
# las = list(dev['label'].values)
# questions.extend(fqs)
# labels.extend(las)

# data = pd.DataFrame({'fq':questions, 'label':labels})
# data.to_csv('{}.csv'.format(os.path.join(root_path, 'data/processed/train')), index=False, sep='\t')



#获取测试集中没有label的样本
test = os.path.join(root_path, 'data/processed/test.csv')
labeled_file = os.path.join(root_path, 'data/processed/test_pre.csv')
data = pd.read_csv(test, sep='\t')
questions = list(data['fq'].values)

test_labeled = pd.read_csv(labeled_file, sep='\t')

labeled = list(test_labeled['fq'].values)

left = []

for ques in questions:
  if ques in labeled:
    continue
  left.append(ques)

left = pd.DataFrame({'fq':left})
left.to_csv('{}.csv'.format(os.path.join(root_path, 'data/processed/left')), index=False, sep='\t')


