""" Define the Transformer model """

# v32
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformer.Layers import EncoderLayer, DecoderLayer
from transformer.SubLayers import PositionwiseFeedForward
import math
import matplotlib.pyplot as plt
from einops import repeat
from transformer.SubLayers import MultiHeadAttention


__author__ = "Yu-Hsiang Huang"
__modified_by__ = "Yudong Zhang"


def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    """For masking out the subsequent info."""
    sz_b, len_s = seq.size()
    subsequent_mask = (
        1 - torch.triu(torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)
    ).bool()
    return subsequent_mask


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer(
            "pos_table", self._get_sinusoid_encoding_table(n_position, d_hid)
        )

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        """Sinusoid position encoding table"""
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [
                position / np.power(10000, 2 * (hid_j // 2) / d_hid)
                for hid_j in range(d_hid)
            ]

        sinusoid_table = np.array(
            [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
        )
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x, passed=True):
        return x + self.pos_table[:, : x.size(-2)].clone().detach()


class Encoder(nn.Module):
    """A encoder model with self attention mechanism."""

    def __init__(
        self,
        d_word_vec,
        n_layers,
        n_head,
        d_k,
        d_v,
        d_model,
        d_inner,
        dropout=0.1,
        n_position=200,
        scale_emb=True,
        n_length=7,
        inoutdim=3,
    ):

        super().__init__()

        self.src_word_emb = nn.Linear(inoutdim, d_model)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList(
            [
                EncoderLayer(
                    d_model,
                    d_inner,
                    n_head,
                    d_k,
                    d_v,
                    n_length=n_length,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm([d_model], eps=1e-6)

        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, src_seq, src_mask, return_attns=False):

        enc_slf_attn_list = []

        # -- Forward
        enc_output = self.src_word_emb(
            src_seq
        )  # change all_zero item [bs, length-1, d_model]
        if self.scale_emb:
            enc_output *= self.d_model**0.5  # amplification
        enc_output = self.dropout(self.position_enc(enc_output, passed=True))
        # enc_output = self.layer_norm(enc_output)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []
        enc_output = self.layer_norm(enc_output)
        if return_attns:
            return enc_output, enc_slf_attn_list
        return (enc_output,)


class Encoder_2(nn.Module):
    """A encoder model with self attention mechanism."""

    def __init__(
        self,
        d_word_vec,
        n_layers,
        n_head,
        d_k,
        d_v,
        d_model,
        d_inner,
        dropout=0.1,
        n_position=200,
        scale_emb=True,
        n_length=7,
        inoutdim=3,
    ):

        super().__init__()

        self.src_word_emb = nn.Linear(inoutdim * n_length, d_model)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList(
            [
                EncoderLayer(
                    d_model,
                    d_inner,
                    n_head,
                    d_k,
                    d_v,
                    n_length=n_length,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm([d_model], eps=1e-6)

        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, src_seq, src_mask, return_attns=False):

        enc_slf_attn_list = []
        src_seq = src_seq.view(
            list(src_seq.size())[:-2] + [-1]
        )  # [bs, num_cand, len_future*featnum]
        # -- Forward
        enc_output = self.src_word_emb(
            src_seq
        )  # change all_zero item [bs, length-1, d_model]
        if self.scale_emb:
            enc_output *= self.d_model**0.5  # amplification
        enc_output = self.dropout(self.position_enc(enc_output, passed=True))
        # enc_output = self.layer_norm(enc_output)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []
        enc_output = self.layer_norm(enc_output)
        if return_attns:
            return enc_output, enc_slf_attn_list
        return (enc_output,)


class Decoder(nn.Module):
    """A decoder model with self attention mechanism."""

    def __init__(
        self,
        d_word_vec,
        n_layers,
        n_head,
        d_k,
        d_v,
        d_model,
        d_inner,
        n_position=200,
        dropout=0.1,
        scale_emb=True,
        n_length=2,
        inoutdim=3,
    ):

        super().__init__()

        self.trg_word_emb = nn.Linear(inoutdim * n_length, d_model)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList(
            [
                DecoderLayer(
                    d_model,
                    d_inner,
                    n_head,
                    d_k,
                    d_v,
                    n_length=n_length,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm([d_model], eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, trg_seq, trg_mask, enc_output, src_mask, return_attns=False):

        # trg_seq [bs, num_cand, len_future, featnum]
        dec_slf_attn_list, dec_enc_attn_list = [], []
        trg_seq = trg_seq.view(
            list(trg_seq.size())[:-2] + [-1]
        )  # [bs, num_cand, len_future*featnum]

        dec_output = self.trg_word_emb(trg_seq)  # change zero_item

        if self.scale_emb:
            dec_output *= self.d_model**0.5
        dec_output = self.dropout(dec_output)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output,
                enc_output,
                slf_attn_mask=trg_mask,
                dec_enc_attn_mask=src_mask,
            )
            dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []
        dec_output = self.layer_norm(dec_output)
        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list

        return (dec_output,)


class Decoder_1(nn.Module):
    """A decoder model with self attention mechanism."""

    def __init__(
        self,
        d_word_vec,
        n_layers,
        n_head,
        d_k,
        d_v,
        d_model,
        d_inner,
        n_position=200,
        dropout=0.1,
        scale_emb=True,
        n_length=2,
        inoutdim=3,
    ):

        super().__init__()

        self.trg_word_emb = nn.Linear(inoutdim, d_model)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList(
            [
                DecoderLayer(
                    d_model,
                    d_inner,
                    n_head,
                    d_k,
                    d_v,
                    n_length=n_length,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm([d_model], eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, trg_seq, trg_mask, enc_output, src_mask, return_attns=False):

        # trg_seq [bs, num_cand, len_future, featnum]
        dec_slf_attn_list, dec_enc_attn_list = [], []
        # trg_seq = trg_seq.view(
        #     list(trg_seq.size())[:-2] + [-1]
        # )  # [bs, num_cand, len_future*featnum]

        dec_output = self.trg_word_emb(trg_seq)  # change zero_item

        if self.scale_emb:
            dec_output *= self.d_model**0.5
        dec_output = self.dropout(dec_output)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output,
                enc_output,
                slf_attn_mask=trg_mask,
                dec_enc_attn_mask=src_mask,
            )
            dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []
        dec_output = self.layer_norm(dec_output)
        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list

        return (dec_output,)


class Pred_reghead(nn.Module):
    def __init__(
        self,
        d_model,
        inoutdim=3,
    ):
        super(Pred_reghead, self).__init__()
        self.reg_mlp = nn.Sequential(
            nn.Linear(d_model, d_model, bias=True),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model // 2, bias=True),
            nn.Linear(d_model // 2, inoutdim, bias=True),
        )
        # self.reg_linear = nn.Linear(n_candi, 1, bias=True)

        self.inoutdim = inoutdim

    def forward(self, x):  # bs,n_length, d_model
        pred = self.reg_mlp(x)  # bs, n_length, inoutdim
        # pred = pred.transpose(-1, -2)  # bs, inoutdim*n_length,candi_num=25
        # pred = self.reg_linear(pred).squeeze(dim=-1)  # bs, inoutdim*n_length
        # pred = pred.view(*pred.shape[0:1], -1, self.inoutdim)  # bs, n_length, inoutdim
        # inoutdim = normed[s_x, s_y, s_size, s_inten,  x, y, size, inten,   abs shiftx, abs shift y, abs dist,     flag]
        pred[..., -1] = torch.sigmoid(pred[..., -1])
        # pred_pos = pred[:,:,:-1].cumsum(dim=-2)
        # pred = torch.cat([pred_pos,pred[:,:,-1:]],-1)

        # cls_h = self.cls_FFN(x).squeeze(dim=-1)
        # conf = self.cls_opt(cls_h)
        return pred  # , cls_h


class pred_biatt_clshead(nn.Module):
    """consider decoder output as key value, consider encoder output as query, and use the attention to do classification"""

    def __init__(
        self,
        n_head,
        d_model,
        d_k,
        d_v,
        dropout=0.1,
        n_length=6,
        out_size=4,
        inoutdim=12,
    ) -> None:
        super().__init__()
        self.cross_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout, n_length=n_length
        )
        self.cls_FFN = nn.Sequential(
            nn.Linear(n_head * n_length, n_head * n_length * 2, bias=True),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(n_head * n_length * 2, n_head * n_length, bias=True),
            nn.Linear(n_head * n_length, n_head * n_length // 2, bias=True),
            nn.Linear(n_head * n_length // 2, 1, bias=True),
        )

        # self.reg_mlp = nn.Sequential(
        #     nn.Linear(d_model, d_model, bias=True),
        #     nn.LayerNorm(d_model),
        #     nn.ReLU(),
        #     nn.Linear(d_model, d_model // 2, bias=True),
        #     nn.Linear(d_model // 2, out_size, bias=True),
        # )
        # self.reg_linear = nn.Linear(n_length, 1, bias=True)
        # self.inoutdim = inoutdim

    def forward(self, enc_output, dec_ouptut):
        enc_output, attn = self.cross_attn(
            q=enc_output, k=dec_ouptut, v=dec_ouptut, mask=None
        )
        # attn.shape = [bs, n_head, q_token_num, k_token_num]
        size_att = [attn.size()[0], -1, attn.size()[-1]]
        attn = attn.view(size_att).transpose(-1, -2)
        prob = self.cls_FFN(attn).squeeze(dim=-1)
        return prob

        # pred = self.reg_mlp(enc_output)  # bs,past_len=6, inoutdim*n_length
        # pred = pred.transpose(-1, -2)  # bs, inoutdim*n_length,past_len=6
        # pred = self.reg_linear(pred).squeeze(dim=-1)  # bs, inoutdim*n_length
        # pred = pred.view(*pred.shape[0:1], -1, self.inoutdim)  # bs, n_length, inoutdim
        # # inoutdim = normed[s_x, s_y, s_size, s_inten,  x, y, size, inten,   abs shiftx, abs shift y, abs dist,     flag]
        # pred[..., -1] = torch.sigmoid(pred[..., -1])

        # return pred, prob


class Transformer(nn.Module):
    """A sequence to sequence model with attention mechanism."""

    def __init__(
        self,
        # src_pad_idx, trg_pad_idx,
        n_passed=7,
        n_future=2,
        n_candi=4,
        d_word_vec=2,
        d_model=512,
        d_inner=2048,
        n_layers=6,
        n_head=8,
        d_k=64,
        d_v=64,
        dropout=0.1,
        n_position=9,
        inoutdim=3,
    ):
        super().__init__()

        scale_emb = True
        self.d_model = d_model

        self.encoder = Encoder(
            n_position=n_position,
            d_word_vec=d_word_vec,
            d_model=d_model,
            d_inner=d_inner,
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
            scale_emb=scale_emb,
            n_length=n_passed,
            inoutdim=inoutdim,
        )

        self.encoder_2 = Encoder_2(
            n_position=n_position,
            d_word_vec=d_word_vec,
            d_model=d_model,
            d_inner=d_inner,
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
            scale_emb=scale_emb,
            n_length=n_future,
            inoutdim=inoutdim,
        )

        self.decoder_1 = Decoder_1(
            n_position=n_position,
            d_word_vec=d_word_vec,
            d_model=d_model,
            d_inner=d_inner,
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
            scale_emb=scale_emb,
            n_length=n_future,
            inoutdim=inoutdim,
        )

        self.vit_token = nn.Parameter(torch.randn(1, n_future, inoutdim))

        # self.decoder_2 = Decoder(
        #     n_position=n_position,
        #     d_word_vec=d_word_vec,
        #     d_model=d_model,
        #     d_inner=d_inner,
        #     n_layers=n_layers,
        #     n_head=n_head,
        #     d_k=d_k,
        #     d_v=d_v,
        #     dropout=dropout,
        #     scale_emb=scale_emb,
        #     n_length=n_future,
        #     inoutdim=inoutdim,
        # )

        self.pred_cls = pred_biatt_clshead(
            n_head=n_head,
            d_model=d_model,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
            n_length=1,
            out_size=n_future * inoutdim,
            inoutdim=inoutdim,
        )

        self.pred_reghead = Pred_reghead(
            d_model=self.d_model,
            inoutdim=inoutdim,
        )
        self.convert_ = nn.Sequential(
            nn.Linear(self.d_model * n_future, self.d_model * n_future, bias=True),
            nn.Linear(self.d_model * n_future, d_model, bias=True),
            nn.LayerNorm(self.d_model),
        )

        assert d_model == d_word_vec
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src_seq, trg_seq):
        # src_seq [bs, len_past, dim]
        enc_output, *_ = self.encoder(src_seq, None, return_attns=True)
        # enc_output [bs, len_past, dim]
        # trg_seq [bs, num_cand, len_future, dim]

        future_tokens = repeat(
            self.vit_token, "a b d -> (m a) b d", m=src_seq.size()[0]
        )
        # [bs, n_future, d_model]
        dec_output, *_ = self.decoder_1(
            trg_seq=future_tokens,
            trg_mask=None,
            enc_output=enc_output,
            src_mask=None,
            return_attns=True,
        )

        pred_shift = self.pred_reghead(dec_output)

        dec_output = dec_output.view(dec_output.size()[0], 1, -1)
        dec_output1 = self.convert_(dec_output)

        dec_output, *_ = self.encoder_2(
            trg_seq,
            None,
            return_attns=True,
        )
        # [bs, len_cand, dim]
        pred_score = self.pred_cls(dec_output1, dec_output)

        return pred_shift, pred_score