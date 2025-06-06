import torch
import torch.nn as nn
from src.module.encoder_decoder import EncoderDecoder
from src.module.output_layer import OutputLayer


class Seq2SeqModel(nn.Module):
    def __init__(self, d_model, h, d_ff, vocab_size, dropout=0.1, max_len=256):
        super(Seq2SeqModel, self).__init__()
        self.encoder_decoder = EncoderDecoder(d_model=d_model, h=h, d_ff=d_ff, vocab_size=vocab_size, dropout=dropout, max_len=max_len)
        self.output_layer = OutputLayer(d_model=d_model, vocab_size=vocab_size, dropout=dropout)

    def forward(self, src, tgt):
        encoder_output = self.encoder_decoder(src, tgt)
        output = self.output_layer(encoder_output)
        return output
    