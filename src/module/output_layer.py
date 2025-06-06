import torch
import torch.nn as nn

class OutputLayer(nn.Module):
    def __init__(self, d_model, vocab_size, dropout=0.1):
        super(OutputLayer, self).__init__()
        self.linear = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(x)
        x = self.linear(x)
        return x