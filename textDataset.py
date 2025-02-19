import pyarrow.parquet as pq
import os
from torch.utils.data import Dataset, Sampler
import torch
import numpy as np
from utils import load_parquet
from tqdm import tqdm
from time import time

class TranslationDataset(Dataset):
    def __init__(self, table, tokenizer, device, pad_idx=1, sos_idx=2, eos_idx=3):
        '''
        It includes tokenizer and vocab, so outputs indices of tokens, not tokens themselves. 
        table : list
        '''
        super().__init__()
        self.table = table
        self.device=device
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

        return torch.tensor(dept, dtype=torch.long, device=self.device), torch.tensor(dest, dtype=torch.long, device=self.device), torch.tensor(ground_truth, dtype=torch.long, device=self.device)
    
def get_and_split_dataset(table, tokenizer, valid_ratio, device):
    n = len(table)
    n_train = n - int(n * valid_ratio)
    print(f"length: {n} / train: {n_train}")
    train_set = TranslationDataset(table[:n_train], tokenizer, device)
    valid_set = TranslationDataset(table[n_train:], tokenizer, device)

    return train_set, valid_set
