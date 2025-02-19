from model import Transformer
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
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
        return [1e-1 * self._d_model * min(pow(self._step_count, -0.5), self._step_count * pow(self.warmup_steps, -1.5))]

def train(h, data_path, ckpt_path):
    print("Model Configuration:")
    print(f"\tNumber of layers: {h.N}")
    print(f"\td_model: {h.d_model}")
    print(f"\tnum_heads: {h.num_heads}")
    print(f"\td_ff: {h.d_ff}")
    print(f"\tDropout: {h.dropout}")
    print(f"\tmax_len: {h.max_len}")

    print("\nTraining configuration: ")
    print(f'\tDevice: {h.device}')
    print(f"\tEpoch: {h.epoch}")
    print(f"\tvalid_ratio: {h.valid_ratio}")
    print(f"\tWarmup steps: {h.warmup_steps}")
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

    print('loading data...')
    table = load_data(data_path)
    if os.path.exists(h.tokenizer_path):
        tokenizer = load_tokenizer(h.tokenizer_path)
    else:
        print('done')
        tokenizer = build_tokenizer(h.tokenizer_path, table, max_len=h.max_len, vocab_size=h.vocab_size)

    train_set, valid_set = get_and_split_dataset(table, tokenizer, h.valid_ratio, h.device)

    optimizer = torch.optim.Adam(transformer.parameters(), betas=(h.beta1, h.beta2), eps=h.eps)
    scheduler = Scheduler(optimizer, h.d_model, h.warmup_steps)
    loss_function = torch.nn.CrossEntropyLoss(label_smoothing=0.1).to(device)

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

    print('data split done')

    '''
    # to start training from in the middle of dataset. There might be better way...
    train_loader_iter = train_loader.__iter__()
    for i in tqdm(range(batch)):
        train_loader_iter.__next__()
    '''

    sw = SummaryWriter("log/")
    
    step = 0
    for e in range(epoch, h.epoch):
        start_time = time()
        print(f'epoch: {e + 1}')

        transformer.train()
        iterator = tqdm(train_set, initial=batch)
        for i, (dept, dest, ground_truth) in enumerate(iterator):
                b_start = time()

                optimizer.zero_grad()
                dept = dept.to(device).unsqueeze(0)
                dest = dest.to(device).unsqueeze(0)
                ground_truth = ground_truth.to(device)

                pred = transformer(dept, dest).squeeze(0)

                loss = loss_function(pred, ground_truth)
                iterator.set_postfix_str(f'train error: {loss.item():.5f}, {pred.max():.3f}, {pred.min():.3f}', refresh=True)
                loss.backward()
                sw.add_scalar("train_err", loss.item(), step) 

                optimizer.step()
                scheduler.step()
                sw.add_scalar("learning_rate", scheduler.get_last_lr()[0], step)

                batch += 1
                step += 1
                if step % 10000 == 100:
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

        transformer.eval()
        valid_error = 0
        print("validation")
        for (dept, dest, ground_truth) in tqdm(valid_set):
            dept = dept.to(device).unsqueeze(0)
            dest = dest.to(device).unsqueeze(0)
            ground_truth = ground_truth.to(device)
            with torch.no_grad():
                pred = transformer(dept, dest).squeeze(0)
            valid_error += loss_function(pred, ground_truth).item()

        end_time = time()
        valid_error = valid_error / len(valid_set)
        print(f'time: {end_time - start_time}, valid error: {valid_error}')
        sw.add_scalar('valid_err', valid_error)
    
    sw.close()
    
def main():
    data_path = '/home/dhhyun/data/train'
    config_path = 'config.json'
    ckpt_path = 'ckpt'
    h = get_config(config_path)
    train(h, data_path, ckpt_path)

    
if __name__ == "__main__":
    main()
