import pyarrow.parquet as pq
import os
from torch.utils.data import Dataset, Sampler
import torch
import numpy as np
from utils import load_parquet

class TranslationDataset(Dataset):
    '''
    It includes tokenizer and vocab, so outputs indices of tokens, not tokens themselves. 
    '''
    def __init__(self, table, tokenizer, max_len):
        super().__init__()
        self.table = table
        self.tokenizer = tokenizer
        self.len = len(table)
        self.max_len = max_len
    
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        dept = self.tokenizer.encode(self.table[idx][0]).ids
        dept.append(3)
        dept = torch.nn.functional.pad(torch.tensor(dept), (0, self.max_len - len(dept)), value=1)
        dest = self.tokenizer.encode(self.table[idx][1]).ids
        dest = torch.nn.functional.pad(torch.tensor(dest), (0, self.max_len - len(dest)), value=1)
        return dept, dest

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
    train_idx = list(range(n_train))
    valid_idx = list(range(n_train, n))
    train_sampler = IdxSampler(train_idx)
    valid_sampler = IdxSampler(valid_idx)

    return dataset, train_sampler, valid_sampler