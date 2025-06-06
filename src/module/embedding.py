import torch
from torch import nn
import math

class TokenEmbedding(nn.Module):
    def __init__(self, d_model, VOCAB_SIZE):
        super().__init__()
        self.embeddings = nn.Embedding(VOCAB_SIZE, d_model) 
        # nn.Embedding is better than using a raw tensor because it registers 
        # the embedding matrix as trainable parameters of the model, and handles 
        # indexing efficiently

    def forward(self, X):
        return self.embeddings(X)
    
class PositionEmbedding(nn.Module):
    def __init__(self, d_model, MAX_LEN):
        super().__init__()
        pos_embedding = torch.zeros(MAX_LEN, d_model)

        position = torch.arange(MAX_LEN).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        # To avoid the use of pow function which is less stable and optimized than an exp we convert 
        # the expression of div_term under an exponential form.
        # This gives us the division terms for each component of the position vectors

        pos_embedding[:, 0::2] = torch.sin(position * div_term) # sin for even dimensions
        pos_embedding[:, 1::2] = torch.cos(position * div_term) # cos for odd dimensions

        pos_embedding = pos_embedding.unsqueeze(0) # We add the batch dimension

        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, x):
        return self.pos_embedding[:, :x.size(1), :] # :x.size(1) is for the right sequence size




if __name__ == "__main__":
    # Example usage
    d_model = 512
    VOCAB_SIZE = 10000
    token_embedding = TokenEmbedding(d_model, VOCAB_SIZE)
    
    batch_size = 10
    seq_length = 20
    token_indices = torch.randint(0, VOCAB_SIZE, (batch_size, seq_length))
    
    output = token_embedding(token_indices)
    print(output.shape)  # Should be (batch_size, seq_length, d_model)
    print(output)  # Output embeddings for the given token indices

    max_len = 256
    position_embedding = PositionEmbedding(d_model, max_len)
    position_output = position_embedding(output)
    print(position_output.shape)  # Should be (1, seq_length, d_model)
    print(position_output)  # Output position embeddings for the given sequence length
    