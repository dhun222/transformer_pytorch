import pyarrow.parquet as pq
import os
from torch.utils.data import Dataset, Sampler
import torch
import numpy as np
from utils import load_parquet
from tqdm import tqdm
from time import time

class TranslationDataset(Dataset):
    '''
    It includes tokenizer and vocab, so outputs indices of tokens, not tokens themselves. 
    '''
    def __init__(self, table, tokenizer, max_len):
        super().__init__()
        '''
        self.table = table
				
        self.len = len(table)
        self.dept = np.empty((self.len, max_len), dtype=np.int16)  # dytpe is uint16 since the vocab size is under ~30000
        self.dest = np.empty((self.len, max_len), dtype=np.int16)
        print(f"memory for data: {self.dept.nbytes + self.dest.nbytes}")

        tok_time = 0
        cat_time = 0

        for i, pair in enumerate(tqdm(table)):
            #tok_start = time()
            dept = tokenizer.encode(pair[0]).ids
            dept = np.array(dept, dtype=np.int16)
            dest = np.array(tokenizer.encode(pair[1]).ids, dtype=np.int16)
            #tok_end = time()

            #tok_time += (tok_end - tok_start)


            #cat_start = time()
            self.dept[i, :] = np.expand_dims(np.pad(dept, (0, max_len - len(dept))), 0)
            self.dept[i, min(len(dept), max_len - 1)] = 3
            self.dest[i, :] = np.expand_dims(np.pad(dest, (0, max_len - len(dest))), 0)
            #cat_end = time()

            #cat_time += (cat_end - cat_start)

        del table

        print('writing files..')
        with open('dept.npy', 'wb') as f:
            np.save(f, self.dept, allow_pickle=False)
        with open('dest.npy', 'wb') as f:
            np.save(f, self.dest, allow_pickle=False)
        '''
        # For fast init, just load saved numpy ndarray containint tokenized indices
        print("loading data...")
        self.dept = torch.from_numpy(np.load("dept.npy"))

        self.dest= torch.from_numpy(np.load("dest.npy"))
        self.len = self.dest.shape[0]

        print(f"data size: {(self.dest.nbytes + self.dept.nbytes) / 1024 / 1024 / 1024}")
        print("done.")

    
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.dept[idx, :].to(torch.int32), self.dest[idx, :].to(torch.long)

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
    print(f"length: {n}/train: {n_train}")
    train_idx = list(range(n_train))
    valid_idx = list(range(n_train, n))
    train_sampler = IdxSampler(train_idx)
    print(len(train_sampler))
    valid_sampler = IdxSampler(valid_idx)

    return dataset, train_sampler, valid_sampler
