import torch
import torch.nn as nn
import math
import pdb

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=10000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        # pe[:, 1::2] = torch.cos(position * div_term)
        # pe = pe.unsqueeze(0).transpose(0, 1)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]  # [WORD_NUM, BATCH, DIM]
        return self.dropout(x)


class PositionalEncoding2(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=5000):
        super(PositionalEncoding2, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        # pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]  # [WORD_NUM, BATCH, DIM]
        return self.dropout(x)




if __name__ == "__main__":
    embedding_dim = 8
    p = PositionalEncoding(d_model=embedding_dim)
    p2 = PositionalEncoding2(d_model=embedding_dim)
    B, W, D = 4,10,embedding_dim
    a = torch.zeros(B,W,D)
    p_a = p(a)
    p2_a = p2(a.transpose(0,1)).transpose(0,1)
    pdb.set_trace()