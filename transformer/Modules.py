import torch
import torch.nn as nn
import torch.nn.functional as F

__author__ = "Yu-Hsiang Huang"


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention"""

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        # mask.shape =[bs, 1, q_token_num, k_token_num]
        attn = torch.matmul(q / self.temperature, k.transpose(-2, -1))
        # attn.shape = [bs, n_head, q_token_num, k_token_num]
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)  # [bs, n_head, q_token_num, dim]

        if mask is not None:  #
            mask = mask[:, :, :, -1:]
            output = output.masked_fill(mask == 0, 0.0)
        return output, attn
