import torch
from torch import nn
from src.module.attention import MultiHeadAttention
from src.module.embedding import TokenEmbedding, PositionEmbedding
from src.module.feedforward import FeedForward


class Transformer(nn.Module):
    def __init__(self, d_model, h, d_ff, VOCAB_SIZE, dropout=0.1, MAX_LEN=256):
        super().__init__()
        self.token_embedding = TokenEmbedding(d_model, VOCAB_SIZE)
        self.pos_embedding = PositionEmbedding(d_model, MAX_LEN)

        self.multi_head = MultiHeadAttention(d_model, h, d_model, MAX_LEN)

        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, X, mask=None):
        embedding = self.token_embedding(X) + self.pos_embedding(X)

        contextual_embedding = self.multi_head(self.norm1(embedding), mask)
        contextual_embedding = self.dropout1(contextual_embedding) + embedding

        output = self.feed_forward(self.norm2(contextual_embedding))
        output = self.dropout2(output) + contextual_embedding

        return output
    
if __name__ == "__main__":
    d_model = 512
    h = 8
    d_ff = 2048
    VOCAB_SIZE = 10000
    dropout = 0.1
    MAX_LEN = 256

    transformer = Transformer(d_model, h, d_ff, VOCAB_SIZE, dropout, MAX_LEN)

    batch_size = 10
    seq_length = 20
    token_indices = torch.randint(0, VOCAB_SIZE, (batch_size, seq_length))

    output = transformer(token_indices)
    print(output.shape)  # Should be (batch_size, seq_length, d_model)
    # print(output)  # Output embeddings for the given token indices

    full_mask = torch.ones(seq_length, seq_length, device=token_indices.device).bool()  
    output_with_mask = transformer(token_indices, full_mask)
    print(output_with_mask.shape)  # Should be (batch_size, seq_length, d_model)
    # print(output_with_mask)  # Output embeddings for the given token indices with mask applied

