from model import Transformer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from utils import build_tokenizer ,load_tokenizer, load_data, get_config
from textDataset import get_and_split_dataset
from time import time
from tqdm import tqdm

class Scheduler(torch.optim.lr_scheduler.LRScheduler):
    def __init__(self, optimzier, d_model, warmup_steps, last_epoch=-1):
        self._d_model = pow(d_model, -0.5) 
        self.warmup_steps = warmup_steps
        super().__init__(optimzier, last_epoch)

    def get_lr(self):
        return [self._d_model * min(pow(self._step_count, -0.5), self._step_count * pow(self.warmup_steps, -1.5))]

def train(h, data_path, ckpt_path):
    print("Model Configuration:")
    print(f"\tNumber of layers: {h.N}")
    print(f"\td_model: {h.d_model}")
    print(f"\tnum_heads: {h.num_heads}")
    print(f"\td_ff: {h.d_ff}")
    print(f"\tDropout: {h.dropout}")

    print("\nTraining configuration: ")
    print(f"\tEpoch: {h.epoch}")
    print(f"\tvalid_ratio: {h.valid_ratio}")
    print(f"\tWarmup steps: {h.warmup_steps}")
    print(f"\tBatch size: {h.batch_size}")
    print(f"\tnum_workers: {h.num_workers}")
    print(f"\tbeta1: {h.beta1}")
    print(f"\tbeta2: {h.beta2}")
    print(f"\teps: {h.eps}\n\n")
    
    device = h.device
    transformer = Transformer(
        N=h.N, 
        d_model=h.d_model, 
        num_heads=h.num_heads, 
        d_ff=h.d_ff, 
        vocab_size=h.vocab_size, 
        dropout=h.dropout, 
        max_len=h.max_len
    ).to(device)
    print('model built')

    if os.path.exists(h.tokenizer_path):
        tokenizer = load_tokenizer(h.tokenizer_path)
    else:
        tokenizer = build_tokenizer(h.tokenizer_path, table, max_len=h.max_len, vocab_size=h.vocab_size)

    print('loading data...')
    table = load_data(data_path)
    print('done')

    optimizer = torch.optim.Adam(transformer.parameters(), betas=(h.beta1, h.beta2), eps=h.eps)
    scheduler = Scheduler(optimizer, h.d_model, h.warmup_steps)

    dataset, train_sampler, valid_sampler = get_and_split_dataset(table, tokenizer, h.max_len, h.valid_ratio)
    
    train_loader = DataLoader(dataset, batch_size=h.batch_size, num_workers=h.num_workers, sampler=train_sampler)
    valid_loader = DataLoader(dataset, batch_size=h.batch_size, num_workers=h.num_workers, sampler=valid_sampler) 
    print('data split done')

    epoch = 0
    batch = 0
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path)
        print('loading ckpt...')
        epoch = ckpt['epoch']
        batch = ckpt['batch']
        transformer.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        print(f'epoch: {epoch}, batch: {batch} / {len(train_loader)}')
    
    # to start training from in the middle of dataset. There might be better way...
    train_loader_iter = train_loader.__iter__()
    for i in tqdm(range(batch)):
        train_loader_iter.__next__()

    loss_function = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    
    step = 0
    for e in range(epoch, h.epoch):
        start_time = time()
        print(f'epoch: {e + 1}')

        transformer.train()
        train_len = len(train_loader)
        b_time = 0
        while True:
            try:
                dept, dest = train_loader_iter.__next__()
                b_start = time()
                print(f"\r", end="")
                progress = int((batch / train_len) * 50)
                for i in range(50):
                    if i < progress:
                        print("|", end="")
                    else:
                        print(" ", end="")
                print(f": {batch} / {train_len}. {b_time}/batch. {(train_len - batch) * b_time}", end="")

                optimizer.zero_grad()
                dept = dept.to(device)
                dest = dest.to(device)

                pred = transformer(dept, dest)
                loss = loss_function(pred.reshape(-1, h.vocab_size), dest.reshape(-1, 1).squeeze(1))
                loss.backward()
                optimizer.step()
                scheduler.step()

                batch += 1
                step += 1
                if step % 10000 == 0:
                    torch.save(
                        {
                            'epoch': e, 
                            'batch': batch, 
                            'optimizer_state_dict': optimizer.state_dict(), 
                            'scheduler_state_dict': scheduler.state_dict(), 
                            'model_state_dict': transformer.state_dict(), 
                        }, 
                        ckpt_path
                    )
                
                b_end = time()
                b_time = b_end - b_start
            except StopIteration:
                print("break")
                break

        transformer.eval()
        valid_error = 0
        print("validation")
        for (dept, dest) in tqdm(valid_loader):
            dept = dept.to(device)
            dest = dest.to(device)
            with torch.no_grad():
                pred =transformer(dept, dest)
                valid_error = loss_function(pred.reshape(-1, h.vocab_size), dest.reshape(-1, 1).squeeze(1)).item()

        end_time = time()
        print(f'time: {end_time - start_time}, valid error: {valid_error / len(valid_sampler)}')

    
def main():
    data_path = '/home/dhhyun/data/train'
    config_path = 'config.json'
    ckpt_path = 'ckpt'
    h = get_config(config_path)
    train(h, data_path, ckpt_path)

    
if __name__ == "__main__":
    main()
