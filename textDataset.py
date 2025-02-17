import pyarrow.parquet as pq
import os
from torch.utils.data import Dataset, Sampler
import torch
import numpy as np
from utils import load_parquet
from tqdm import tqdm
from time import time

class TranslationDataset(Dataset):
    def __init__(self, table, tokenizer, max_len, pad_idx=1, sos_idx=2, eos_idx=3):
        '''
        It includes tokenizer and vocab, so outputs indices of tokens, not tokens themselves. 
        table : list
        '''
        super().__init__()
        self.table = table
        self.max_len = max_len
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.tokenizer = tokenizer
				
        self.len = len(table)

    
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        '''
        model's embedding layer only takes long, int dtypes as input
        torch.nn.CrossEntropyLoss only takes long tensor as input
        dept: for encoder
        dest: for decoder
        ground_truth: for loss
        '''
        dept = self.tokenizer.encode(self.table[idx][0]).ids
        dept.append(self.eos_idx)
        dest = self.tokenizer.encode(self.table[idx][1]).ids
        ground_truth = list(dest)
        dest.insert(0, self.sos_idx)
        ground_truth.append(self.eos_idx)

        return self._pad(dept), self._pad(dest), self._pad(ground_truth)
    
    def _pad(self, x):
        # pad data into (max_len - 1) length
        return torch.nn.functional.pad(
            torch.tensor(x, dtype=torch.long), 
            (0, self.max_len - len(x)), 
            value=self.pad_idx
        )

class IdxSampler(Sampler):
    def __init__(self, idx):
        self.idx = idx

    def __iter__(self):
        return iter(self.idx)
    
    def __len__(self):
        return len(self.idx)
    
    def set_start(self, start):
        temp = self.idx
        self.idx = self.idx[start:]
        self.idx.extend(temp[:start])

def get_and_split_dataset(table, tokenizer, max_len, valid_ratio):

    dataset = TranslationDataset(table, tokenizer, max_len)

    n = len(dataset)
    n_train = n - int(n * valid_ratio)
    print(f"length: {n} / train: {n_train}")
    train_idx = list(range(n_train))
    valid_idx = list(range(n_train, n))
    train_sampler = IdxSampler(train_idx)
    valid_sampler = IdxSampler(valid_idx)

    return dataset, train_sampler, valid_sampler
