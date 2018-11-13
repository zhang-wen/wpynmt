from __future__ import division
import torch as tc
import torch.nn as nn
import wargs
from gru import LGRU, TGRU
from tools.utils import *

'''
    Bi-directional Transition Gated Recurrent Unit network encoder
    input args:
        src_emb:        class WordEmbedding
        enc_hid_size:   the size of TGRU hidden state
        n_layers:       layer nubmer of encoder
'''
class StackedTransEncoder(nn.Module):

    def __init__(self,
                 src_emb,
                 enc_hid_size=512,
                 rnn_dropout=0.3,
                 n_layers=3,
                 prefix='TGRU_Encoder', **kwargs):

        super(StackedTransEncoder, self).__init__()

        self.word_emb = src_emb
        n_embed = src_emb.n_embed
        self.enc_hid_size = enc_hid_size
        f = lambda name: str_cat(prefix, name)  # return 'Encoder_' + parameters name

        self.f_lgru = LGRU(n_embed, self.enc_hid_size, dropout_prob=rnn_dropout,
                           prefix=f('{}_{}_{}'.format(prefix, 'LF', 0)))
        self.b_lgru = LGRU(n_embed, self.enc_hid_size, dropout_prob=rnn_dropout,
                           prefix=f('{}_{}_{}'.format(prefix, 'LB', 0)))
        self.f_tgrus = nn.ModuleList( [ TGRU(self.enc_hid_size, dropout_prob=rnn_dropout,
                                             prefix=f('{}_{}_{}'.format(prefix, 'TF', n)))
                                       for n in range(n_layers - 1) ] )
        self.b_tgrus = nn.ModuleList( [ TGRU(self.enc_hid_size, dropout_prob=rnn_dropout,
                                             prefix=f('{}_{}_{}'.format(prefix, 'TB', n)))
                                       for n in range(n_layers - 1) ] )

        self.n_layers = n_layers

    def forward(self, xs, xs_mask=None, h0=None):

        batch_size, max_L = xs.size(0), xs.size(1)
        if xs.dim() == 3: xs_e = xs
        else: x_w_e, xs_e = self.word_emb(xs)

        r_anns, l_anns = [], []
        f_h = b_h = h0 if h0 else tc.zeros(batch_size, self.enc_hid_size, requires_grad=False)
        if wargs.gpu_id is not None: f_h, b_h = f_h.cuda(), b_h.cuda()
        for f_idx in range(max_L):
            # (batch_size, n_embed)
            b_idx = max_L - f_idx - 1
            f_inp, b_inp = xs_e[:, f_idx, :], xs_e[:, b_idx, :]
            f_mask = xs_mask[:, f_idx] if xs_mask is not None else None
            b_mask = xs_mask[:, b_idx] if xs_mask is not None else None
            f_h_jk = self.f_lgru(f_inp, f_h, f_mask)
            b_h_jk = self.b_lgru(b_inp, b_h, b_mask)
            for layer_idx in range(self.n_layers - 1):
                f_h_jk = self.f_tgrus[layer_idx](f_h_jk, f_mask)
                b_h_jk = self.b_tgrus[layer_idx](b_h_jk, b_mask)
            f_h, b_h = f_h_jk, b_h_jk
            r_anns.append(f_h_jk)
            l_anns.append(b_h_jk)
        r_anns, l_anns = tc.stack(r_anns, dim=1), tc.stack(l_anns[::-1], dim=1)

        annotations = tc.cat([r_anns, l_anns], dim=-1)   # (batch_size, max_L, 2*enc_hid_size)

        return annotations * xs_mask[:, :, None]

