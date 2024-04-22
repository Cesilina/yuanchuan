import sys
import os
import time

import numpy as np
import pandas as pd
import faiss
from faiss import normalize_L2

sys.path.append('..')
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(BASE_DIR)
import config
import logging


class HNSW(object):
    '''提取公共的代码
    只需要传入相应的corpus_sentence,corpus_embedding,以及data['starter', 'response']
    以及相应的变量ef、M、embedding_dim、
    model_path和data_path
    '''

    def __init__(self,
                 corpus_embeddings,
                 data,
                 ef=config.ef_construction,
                 M=config.M,
                 embedding_dim=300,
                 model_path=None
                 ):
        # self.corpus_sentences = corpus_sentences
        self.corpus_embedding = corpus_embeddings
        self.embedding_dim = embedding_dim
        self.data = data

        # if model_path and os.path.exists(model_path):
        #     # 加载
        #     self.index = self.load_hnsw(model_path)
        # elif model_path:
            # 训练
        self.index = self.build_hnsw(model_path, ef=ef, m=M)
        # else:
            # logging.error('No existing model and no building data provided.')

    def evaluate(self, vecs):
        '''
        @description: 评估模型。
        @param {type} text: The query.
        @return {type} None
        '''
        logging.info('Evaluating.')
        nq, d = vecs.shape
        t0 = time.time()
        D, I = self.index.search(vecs, 1)
        t1 = time.time()

        missing_rate = (I == -1).sum() / float(nq)
        recall_at_1 = (I == np.arange(nq)).sum() / float(nq)
        print("\t %7.3f ms per query, R@1 %.4f, missing rate %.4f" % (
            (t1 - t0) * 1000.0 / nq, recall_at_1, missing_rate))

    def build_hnsw(self, to_file, ef=2000, m=64):
        '''
        @description: 训练hnsw模型
        @param {type}
        to_file： 模型保存目录
        @return:
        '''
        logging.info('Building hnsw index.')

        self.corpus_embedding = self.corpus_embedding.astype('float32')
        dim = self.corpus_embedding.shape[1]
        measure = faiss.METRIC_INNER_PRODUCT

        # Declaring index
        index = faiss.IndexHNSWFlat(dim, m, measure)  # build the index
        res = faiss.StandardGpuResources()  # use a single GPU
        faiss.index_cpu_to_gpu(res, 1, index)  # make it a GPU index
        index.hnsw.efConstruction = ef
        print("add")
        index.verbose = True  # to see progress
        print('xb: ', self.corpus_embedding.shape)

        print('dtype: ', self.corpus_embedding.dtype)

        logging.info('normalize corpus_embedding')
        normalize_L2(self.corpus_embedding)

        index.add(self.corpus_embedding)  # add vectors to the index
        print("total: ", index.ntotal)
        faiss.write_index(index, to_file)
        return index

    def load_hnsw(self, model_path):
        '''
        @description: 加载训练好的hnsw模型
        @param {type}
        model_path： 模型保存的目录
        @return: hnsw 模型
        '''
        logging.info(f'Loading hnsw index from {model_path}.')
        hnsw = faiss.read_index(model_path)
        return hnsw

    def search(self, text_vec, k=5):
        '''
        @description: 通过hnsw 检索
        @param {type}
        text: 检索句子
        k: 检索返回的数量
        @return: DataFrame contianing the customer input, assistance response
                and the distance to the query.
        '''
        logging.info(f'Searching .')
        text_vec = text_vec.astype('float32')
        normalize_L2(text_vec)
        # vecs is a n2-by-d matrix with query vectors
        k = k  # we want 4 similar vectors
        D, I = self.index.search(text_vec, k)

        return pd.concat(
            (self.data.iloc[I[0]]['fq'].reset_index(),
             self.data.iloc[I[0]]['label'].reset_index(drop=True),
             pd.DataFrame(D.reshape(-1, 1), columns=['q_distance'])),
            axis=1)
