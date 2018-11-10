''' Define the Transformer model '''
import math
import torch as tc
import torch.nn as nn
import torch.nn.init as init
import numpy as np

import wargs
from tools.utils import *
from models.losser import *
from models.embedding import WordEmbedding

__author__ = "Yu-Hsiang Huang"
np.set_printoptions(threshold=np.nan)

class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(self, n_src_vocab, n_tgt_vocab, n_max_seq, n_layers=6, n_head=8,
                 d_word_vec=512, d_model=512, d_inner_hid=1024, dropout=0.1,
                 proj_share_weight=True, embs_share_weight=True):

        wlog('Transformer Model ========================= ')
        wlog('\tn_src_vocab:        {}'.format(n_src_vocab))
        wlog('\tn_trg_vocab:        {}'.format(n_tgt_vocab))
        wlog('\tn_max_seq:          {}'.format(n_max_seq))
        wlog('\tn_layers:           {}'.format(n_layers))
        wlog('\tn_head:             {}'.format(n_head))
        wlog('\td_word_vec:         {}'.format(d_word_vec))
        wlog('\td_model:            {}'.format(d_model))
        wlog('\td_inner_hid:        {}'.format(d_inner_hid))
        wlog('\tdropout:            {}'.format(dropout))
        wlog('\tproj_share_weight:  {}'.format(proj_share_weight))
        wlog('\tembs_share_weight:  {}'.format(embs_share_weight))

        super(Transformer, self).__init__()
        self.encoder = Encoder(n_src_vocab, n_max_seq, n_layers=n_layers, n_head=n_head,
                               d_word_vec=d_word_vec, d_model=d_model,
                               d_inner_hid=d_inner_hid, dropout=dropout)
        self.decoder = Decoder(n_tgt_vocab, n_max_seq, n_layers=n_layers, n_head=n_head,
                               d_word_vec=d_word_vec, d_model=d_model,
                               d_inner_hid=d_inner_hid, dropout=dropout,
                               proj_share_weight=proj_share_weight)

        assert d_model == d_word_vec, 'To facilitate the residual connections, \
                the dimensions of all module output shall be the same.'
        if embs_share_weight is True:
            # Share the weight matrix between src and tgt word embeddings
            assert n_src_vocab == n_tgt_vocab, \
            "To share word embedding table, the vocabulary size of src and tgt shall be the same."
            self.encoder.src_word_emb.weight = self.decoder.tgt_word_emb.weight

    def get_trainable_parameters(self):
        ''' Avoid updating the position encoding '''
        enc_freezed_param_ids = set(map(id, self.encoder.position_enc.parameters()))
        dec_freezed_param_ids = set(map(id, self.decoder.position_enc.parameters()))
        freezed_param_ids = enc_freezed_param_ids | dec_freezed_param_ids
        return ((n, p) for (n, p) in self.named_parameters() if id(p) not in freezed_param_ids)

    def forward(self, src, tgt):

        src_seq, src_pos = src
        tgt_seq, tgt_pos = tgt

        tgt_seq = tgt_seq[:, :-1]
        tgt_pos = tgt_pos[:, :-1]

        enc_outputs, enc_slf_attn, enc_slf_one_attn = self.encoder(src_seq, src_pos)
        enc_output = enc_outputs[-1]
        dec_output, dec_slf_attns, dec_enc_attns, dec_enc_one_attn = \
                self.decoder(tgt_seq, tgt_pos, src_seq, enc_output)

        return dec_output

