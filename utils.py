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
    
    table = []
    for file in file_list:
        _table = load_parquet(os.path.join(path, file))
        table.extend(_table)
    return table


def load_parquet(path):
    '''
    Input
        path: path for parquet file
    output
        table: list of tuples
    '''
    _table = pq.read_table(path)["translation"]
    table = _table.to_pylist() 
    del _table
    keys = list(table[0].keys())
    for i, pair in enumerate(table):
        table[i] = (pair[keys[0]], pair[keys[1]])

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
    tokenizer.post_processor = TemplateProcessing(single="[sos] $0", special_tokens=[('[sos]', 2)])
    tokenizer.enable_truncation(max_length=max_len)
    tokenizer.save(path)
    print("done")
    
    return tokenizer
