import torch
from torch import nn
from src.module.attention import MultiHeadAttention, CrossAttention
from src.module.embedding import TokenEmbedding, PositionEmbedding
from src.module.feedforward import FeedForward
from src.module.transformer import Transformer

class EncoderDecoder(nn.Module):
    def __init__(self, d_model, h, d_ff, vocab_size, dropout=0.1, max_len=256):
        super().__init__()
        self.encoder = Transformer(d_model, h, d_ff, vocab_size, dropout, max_len)

        self.token_embedding = TokenEmbedding(d_model, vocab_size)
        self.pos_embedding = PositionEmbedding(d_model, max_len)

        self.masked_layer = MultiHeadAttention(d_model, h, None, max_len)

        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.cross_attention = CrossAttention(d_model, h, None, max_len)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)

    
    def forward(self, X_encoder, X_decoder):
        encoder_output = self.encoder(X_encoder)

        decoder_embedding = self.token_embedding(X_decoder) + self.pos_embedding(X_decoder)
        masked_output = self.masked_layer(self.norm1(decoder_embedding), mask='causal')
        masked_output = self.dropout1(masked_output) + decoder_embedding

        cross_output = self.cross_attention(encoder_output, self.norm2(masked_output))
        cross_output = self.dropout2(cross_output) + masked_output

        output = self.feed_forward(self.norm3(cross_output))
        output = self.dropout3(output) + cross_output



        return output
    
if __name__ == "__main__":
    d_model = 512
    h = 8
    d_ff = 2048
    vocab_size = 10000
    dropout = 0.1
    max_len = 256

    encoder_decoder = EncoderDecoder(d_model, h, d_ff, vocab_size, dropout, max_len)

    batch_size = 10
    seq_length_enc = 20
    seq_length_dec = 15
    token_indices_enc = torch.randint(0, vocab_size, (batch_size, seq_length_enc))
    token_indices_dec = torch.randint(0, vocab_size, (batch_size, seq_length_dec))

    output = encoder_decoder(token_indices_enc, token_indices_dec)
    print(output.shape)  # Should be (batch_size, seq_length_dec, d_model)
    # print(output)  # Output embeddings for the given token indices in the decoder