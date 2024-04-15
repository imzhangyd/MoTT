""" Define the Layers """

import torch.nn as nn

# import torch
from transformer.SubLayers import MultiHeadAttention, PositionwiseFeedForward


__author__ = "Yu-Hsiang Huang"


class EncoderLayer(nn.Module):
    """Compose with two layers"""

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.layernorm = nn.LayerNorm([d_model])
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout
        )
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout
        )

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output = self.layernorm(enc_input)
        enc_output, enc_slf_attn = self.slf_attn(
            enc_output, enc_output, enc_output, mask=slf_attn_mask
        )
        enc_output = enc_output + enc_input

        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class DecoderLayer(nn.Module):
    """Compose with three layers"""

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.layernorm1 = nn.LayerNorm([d_model])
        self.layernorm2 = nn.LayerNorm([d_model])
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout
        )
        self.enc_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout
        )
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout
        )

    def forward(
        self, dec_input, enc_output, slf_attn_mask=None, dec_enc_attn_mask=None
    ):
        dec_output = self.layernorm1(dec_input)
        dec_output, dec_slf_attn = self.slf_attn(
            dec_output, dec_output, dec_output, mask=slf_attn_mask
        )
        dec_output = dec_output + dec_input

        dec_output_2 = self.layernorm2(dec_output)
        dec_output_2, dec_enc_attn = self.enc_attn(
            dec_output_2, enc_output, enc_output, mask=dec_enc_attn_mask
        )

        dec_output = dec_output_2 + dec_output

        dec_output = self.pos_ffn(dec_output)
        return dec_output, dec_slf_attn, dec_enc_attn
