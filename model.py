import torch
from torch import nn
from utils import get_config

class PositionEncoder(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500, dropout=0.1):
        super().__init__()
        x = torch.arange(0, max_len).unsqueeze(1)
        x = x * (1 / 10000**(torch.arange(0, d_model, 2) / d_model))
        pe = torch.empty(max_len, d_model)
        pe[:, 0::2] = torch.sin(x)
        pe[:, 1::2] = torch.cos(x)
        self.register_buffer("pe", pe)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(x + self.pe[:x.shape[-2], :])

class InputEmbedding(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.root_d_model = pow(d_model, 0.5)
    
    def forward(self, x):
        return self.embedding(x) * self.root_d_model

class MultiHeadAttention(nn.Module):
    '''
    Multihead attention module with masking
    Key and Value are same
    '''
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, 'd_model has to be multiple of num_head'
        self.num_heads = num_heads
        self.d_model = d_model
        self.root_d_k = d_model**(1/2)
        self.d_h = self.d_model // self.num_heads
        self.Q_heads = nn.Linear(d_model ,d_model)
        self.K_heads = nn.Linear(d_model, d_model)
        self.V_heads = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        # linear projection
        # [b, T, d_model]
        Q = self.Q_heads(Q)
        K = self.K_heads(K)
        V = self.V_heads(V)

        # Calculate score
        # Q [B, num_heads, T, d_h]
        # Transpose is for data align for reshape function
        Q = torch.transpose(Q, -1, -2).reshape((-1, self.num_heads, Q.shape[-2], self.d_h))
        # K [B, num_heads, d_h, T]
        K = torch.transpose(torch.transpose(K, -1, -2).reshape((-1, self.num_heads, K.shape[-2], self.d_h)), -1, -2)
        score = torch.softmax(torch.matmul(Q, K) / self.root_d_k, dim=-1)

        # Masking
        # Elements at same position with elements which have True in mask tensor will be masked out
        if mask is not None:
            mask = mask.to(score.device)
            score = score.masked_fill(mask, -1e9)

        # Attention
        att = torch.matmul(score, V.reshape((-1, self.num_heads, V.shape[-2], self.d_h)))

        att = att.reshape(-1, att.shape[-2], self.d_model).squeeze(0)

        # Fianl linear projection
        att = self.W_o(att)

        # Dropout
        att = self.dropout(att)

        return att

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(d_model, d_ff), 
            nn.ReLU(), 
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.dropout(self.layer(x))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention_layer = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model) 
        self.ff_layer = FeedForward(d_model, d_ff, dropout=dropout) 
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, emb):
        x = self.norm1(emb + self.attention_layer(emb, emb, emb))
        return self.norm2(x + self.ff_layer(x))

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.masked_attention_layer = MultiHeadAttention(d_model, num_heads, dropout=dropout) 
        self.norm1 = nn.LayerNorm(d_model)
        self.attention_layer = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff_layer = FeedForward(d_model, d_ff)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, emb, context):
        mask = self.get_causal_mask(emb.shape[-2])
        x = self.norm1(emb + self.masked_attention_layer(emb, emb, emb, mask))
        x = self.norm2(x + self.attention_layer(x, context, context))
        return self.norm3(x + self.ff_layer(x))     
    
    def get_causal_mask(self, size):
        mask = torch.ones((size, size))
        mask = torch.tril(mask) == 0

        return mask


class Generator(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)
        self.softmax = nn.Softmax(dim=-1)
        self.inference = False


    def forward(self, x):
        if self.inference:
            return self.softmax(self.linear(x[..., -1, :]))
        else:
            return self.softmax(self.linear(x))


class Transformer(nn.Module):
    def __init__(self, N, d_model, num_heads, d_ff, vocab_size, dropout=0.1, max_len=500):
        '''
        N : number of encoder/decoder layers
        d_model : dimensions of hidden layers
        num_heads : number of attention heads
        d_ff : dimensions of feedforward layer
        tokenizer_path : path to pretrained tokenizer
        '''
        super().__init__()
        self.input_embedding = InputEmbedding(d_model, vocab_size)
        self.positional_encoder = PositionEncoder(d_model, max_len, dropout=dropout)
        
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for i in range(N):
            self.encoder.append(EncoderLayer(d_model, num_heads, d_ff, dropout=dropout))
            self.decoder.append(DecoderLayer(d_model, num_heads, d_ff, dropout=dropout))
        
        self.generator = Generator(d_model, vocab_size)
    
    def forward(self, encoder_input, decoder_input):
        encoder_emb = self.input_embedding(encoder_input)
        context = self.positional_encoder(encoder_emb)
        for layer in self.encoder:
            context = layer(context)

        decoder_emb = self.input_embedding(decoder_input)
        out = self.positional_encoder(decoder_emb)
        for layer in self.decoder:
            out = layer(out, context)
        return self.generator(out)
    
class Translator(nn.Module):
    def __init__(self, h):
        super().__init__()
        transformer = Transformer(
            N=h.N, 
            d_model=h.d_model, 
            num_heads=h.num_heads, 
            d_ff=h.d_ff, 
            dropout=h.droput, 
            max_len=h.max_len
        )
        
    def forward(self, x):
        return
