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

        self.word_emb = src_emb.we
        n_embed = src_emb.n_embed
        self.enc_hid_size = enc_hid_size
        f = lambda name: str_cat(prefix, name)  # return 'Encoder_' + parameters name

        wlog(n_layers)
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

        max_L, batch_size = xs.size(0), xs.size(1)
        xs_e = xs if xs.dim() == 3 else self.word_emb(xs)

        r_annotations, l_annotations = [], []
        f_h = b_h = h0 if h0 else tc.zeros(batch_size, self.enc_hid_size, requires_grad=False)
        if wargs.gpu_id is not None: f_h, b_h = f_h.cuda(), b_h.cuda()
        for f_idx in range(max_L):
            # (batch_size, n_embed)
            b_idx = max_L - f_idx - 1
            f_h_jk = self.f_lgru(xs_e[f_idx], f_h, xs_mask[f_idx] if xs_mask is not None else None)
            b_h_jk = self.b_lgru(xs_e[b_idx], b_h, xs_mask[b_idx] if xs_mask is not None else None)
            for layer_idx in range(self.n_layers - 1):
                f_h_jk = self.f_tgrus[layer_idx](f_h_jk, xs_mask[f_idx])
                b_h_jk = self.b_tgrus[layer_idx](b_h_jk, xs_mask[b_idx])
            f_h, b_h = f_h_jk, b_h_jk
            r_annotations.append(f_h_jk)
            l_annotations.append(b_h_jk)
        r_annotations, l_annotations = tc.stack(r_annotations), tc.stack(l_annotations[::-1])

        annotations = tc.cat([r_annotations, l_annotations], dim=-1)   # (max_L, batch_size, 2*enc_hid_size)

        return annotations * xs_mask[:, :, None]

