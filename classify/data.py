from torch.utils.data import Dataset
import pandas as pd
import torch
def rpad(array, n=300):
    """Right padding."""
    current_len = len(array)
    if current_len >= n:
        return array[: n]
    extra = n - current_len
    return array + ([0] * extra)

class SSTDataset(Dataset):
  def __init__(self, data_path, tokenizer, maxlen):
        self.corpus, data, labels = self.get_all_corpus_sentence(data_path)  # get_all_corpus_sentence(data_path)(-->闲聊数据集)
        self.data = data
        self.labels = labels
        self.tokenizer = tokenizer
        self.maxlen = maxlen

  def text_to_id(self, source):
      sample = rpad(self.tokenizer.encode("[CLS] " + source + " [SEP]"))
      return sample

  def __len__(self):
      return len(self.data)

  def __getitem__(self, idx):
      x,y = self.text_to_id(self.data[idx]), self.labels[idx]
      x = torch.tensor(x)
      return x,y

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
      return corpus, content, label_list

