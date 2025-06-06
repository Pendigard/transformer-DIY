import torch
from torch import nn
import math
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h, d_v=None, MAX_LENGTH=256):
        super().__init__()
        assert d_model % h == 0 # The dimension of the input embeddings must be divisible by h
        self.h = h
        self.d_k = d_model // h
        self.d_v = d_v
        if d_v == None:
            self.d_v = self.d_k
        self.MAX_LENGTH = MAX_LENGTH
        self.W_Q = nn.Linear(d_model, h * self.d_k)
        self.W_K = nn.Linear(d_model, h * self.d_k)
        self.W_V = nn.Linear(d_model, h * self.d_v)
        self.W_O = nn.Linear(h * self.d_v, d_model)

    def forward(self, X, mask=None):
        """
        X: (batch_size, seq_len, d_model)
        mask: (batch_size, seq_len) or None
        """
        L = X.shape[1]
        if mask is None:
            mask = torch.ones((L, L), device=X.device).bool()  # (L, L)
            mask = mask.unsqueeze(0).unsqueeze(1)  # Add batch and head dimensions
        elif mask == 'causal':
            mask = torch.tril(torch.ones((L, L), device=X.device)).bool()  # (L, L)
            mask = mask.unsqueeze(0).unsqueeze(1)
        else:
            mask = mask.unsqueeze(0).unsqueeze(1)

        Q = self.W_Q(X).reshape(X.shape[0], L, self.h, self.d_k).transpose(1, 2)
        # Reshape to (batch_size, seq_len, h, d_k) and transpose to (batch_size, h, seq_len, d_k)
        K = self.W_K(X).reshape(X.shape[0], L, self.h, self.d_k).transpose(1, 2)
        V = self.W_V(X).reshape(X.shape[0], L, self.h, self.d_v).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k) # -2 and -1 to invert the two last dimensions, the first dimension being the batch
        scores = scores.masked_fill(mask == 0, -1e9) # Only scores that are not masked will be considered
        A = torch.softmax(scores , dim = -1) # dim=-1 because we apply the softmax for each token
        output = A @ V  # (batch_size, h, seq_len, d_v)

        output = output.transpose(1, 2).reshape(X.shape[0], X.shape[1], self.h * self.d_v)
        return self.W_O(output)  # Apply the final linear transformation to combine the heads
    
class CrossAttention(MultiHeadAttention):
    def __init__(self, d_model, h, d_v=None, MAX_LENGTH=256):
        super().__init__(d_model, h, d_v, MAX_LENGTH)

    def forward(self, X_encoder, X_decoder):
        batch_size, L_dec, _ = X_decoder.shape
        L_enc = X_encoder.shape[1]

        Q = self.W_Q(X_decoder).reshape(batch_size, L_dec, self.h, self.d_k).transpose(1, 2)
        K = self.W_K(X_encoder).reshape(batch_size, L_enc, self.h, self.d_k).transpose(1, 2)
        V = self.W_V(X_encoder).reshape(batch_size, L_enc, self.h, self.d_v).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
        A = torch.softmax(scores, dim=-1)
        output = A @ V

        output = output.transpose(1, 2).reshape(batch_size, L_dec, self.h * self.d_v)
        return self.W_O(output)

if __name__ == "__main__":
    attention = MultiHeadAttention(512, 8)
    X = torch.randn(10, 256, 512)  # Batch size of 10, sequence length of 256, embedding dimension of 512
    output_decoder = attention(X, mask='causal')  # Causal mask for decoder
    print(output_decoder.shape)

    output_encoder = attention(X)
    print(output_encoder.shape)

    # Example for CrossAttention
    X_encoder = torch.randn(10, 256, 512)  # Encoder output
    X_decoder = torch.randn(10, 128, 512)  # Decoder input
    cross_attention = CrossAttention(512, 8)
    output_cross = cross_attention(X_encoder, X_decoder)
    print(output_cross.shape)  # Should be (10, 128, 512)
