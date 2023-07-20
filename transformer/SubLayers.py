''' Define the sublayers in encoder/decoder layer '''
from cv2 import dnn_Model
import numpy as np
from torch import unsqueeze
import torch.nn as nn
import torch.nn.functional as F
from transformer.Modules import ScaledDotProductAttention

__author__ = "Yu-Hsiang Huang"

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1,n_length=2):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)



    def forward(self, q, k, v, mask=None):
        
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        
        qsize_list = list(q.size()[:-1])
        qsize_list.append(n_head)
        qsize_list.append(d_k)
        q = self.w_qs(q).view(qsize_list)
        ksize_list = list(k.size()[:-1])
        ksize_list.append(n_head)
        ksize_list.append(d_k)
        k = self.w_ks(k).view(ksize_list)
        vsize_list = list(v.size()[:-1])
        vsize_list.append(n_head)
        vsize_list.append(d_v)
        v = self.w_vs(v).view(vsize_list)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(-3, -2), k.transpose(-3, -2), v.transpose(-3, -2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        if len(q.shape) > len(k.shape):
            k = k.unsqueeze(1)
            v = v.unsqueeze(1)
        q, attn = self.attention(q, k, v, mask=mask) #


        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        
        q = q.transpose(-3, -2)
        qsize_list = list(q.size()[:-2])
        qsize_list.append(-1)
        q = q.contiguous().view(qsize_list)
        q = self.dropout(self.fc(q))


        return q, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, n_length,dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm([d_in], eps=1e-6)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):

        residual = x
        x = self.layer_norm(x) #  pre LN
        x = F.relu(self.w_1(x))
        x = self.dropout1(x)
        x = self.w_2(x)
        x = self.dropout2(x)
        x += residual

        return x
