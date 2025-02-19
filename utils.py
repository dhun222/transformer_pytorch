import os
import pyarrow.parquet as pq
from io import open
import json
from tokenizers import Tokenizer 
from tokenizers.trainers import BpeTrainer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def get_config(path):
    with open(path) as f:
        data = f.read()

    h_json = json.loads(data)
    h = AttrDict(h_json)
    return h

def load_data(path):
    file_list = []
    for file in os.listdir(path):
        if file.endswith('.parquet'):
            file_list.append(file)
    
    print(f'{len(file_list)} .parquet files are found..')
    
    table = []
    for file in file_list:
        _table = load_parquet(os.path.join(path, file))
        table.extend(_table)
    return table


def load_parquet(path, omit_len=500):
    '''
    Input
        path: path for parquet file
        omit_len: If the length of string is larger than omit_len, the string is omitted. 
    output
        table: list of tuples
    '''
    _table = pq.read_table(path)["translation"]
    _table = _table.to_pylist() 
    keys = list(_table[0].keys())
    table = []
    for i, pair in enumerate(_table):
        if len(pair[keys[0]]) < omit_len and len(pair[keys[1]]) < omit_len:
            table.append((pair[keys[0]], pair[keys[1]]))
    del _table

    return table

def corpus_from_table(table):
    for pair in table:
        for seq in pair:
            yield seq


def load_tokenizer(path):
    print("loading tokenizer from file...")
    tokenizer = Tokenizer(BPE()).from_file(path)
    print("done")

    return tokenizer

def build_tokenizer(path, data, vocab_size=15000, max_len=500):
    print("building tokenizer from corpus...")
    tokenizer = Tokenizer(BPE(unk_token='[unk]'))
    trainer = BpeTrainer(vocab_size=vocab_size, show_progress=True, 
                            special_tokens=['[unk]', '[pad]', '[sos]', '[eos]'])
    corpus = corpus_from_table(data)
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train_from_iterator(corpus, trainer)
    del corpus
    tokenizer.enable_truncation(max_length=max_len - 1)
    tokenizer.save(path)
    print("done")
    
    return tokenizer
