''' Define the Transformer model '''
import math
import torch as tc
import torch.nn as nn
import torch.nn.init as init
import numpy as np

import wargs
from tools.utils import *
from models.losser import *

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

class MultiHeadAttention(nn.Module):
    '''
        Multi-Head Attention module from <Attention is All You Need>
        Args:
            n_head(int):    number of parallel heads.
            d_model(int):   the dimension of keys/values/queries in this MultiHeadAttention
                d_model % n_head == 0
            d_k(int):       the dimension of queries and keys
            d_v(int):       the dimension of values
    '''
    def __init__(self, d_model, n_head, d_k, d_v, dropout=0.1):

        super(MultiHeadAttention, self).__init__()

        assert d_model % n_head == 0, 'd_model {} divided by n_head {}.'.format(d_model, n_head)
        self.d_model, self.n_head, self.d_k, self.d_v = d_model, n_head, d_k, d_v

        self.w_q = nn.Parameter(tc.FloatTensor(n_head, d_model, d_k))
        self.w_k = nn.Parameter(tc.FloatTensor(n_head, d_model, d_k))
        self.w_v = nn.Parameter(tc.FloatTensor(n_head, d_model, d_v))
        init.xavier_normal(self.w_q)
        init.xavier_normal(self.w_k)
        init.xavier_normal(self.w_v)

        self.mSoftMax = MaskSoftmax()
        self.layer_norm = Layer_Norm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.temper = np.power(d_model, 0.5)

        #self.proj = nn.Linear(n_head*d_v, d_model)
        self.proj = XavierLinear(n_head*d_v, d_model)

    def forward(self, q, k, v, attn_mask=None):

        B_q, L_q, d_model_q = q.size()
        B_k, L_k, d_model_k = k.size()
        B_v, L_v, d_model_v = v.size()
        assert B_k == B_v and L_k == L_v and d_model_k == d_model_v == self.d_model
        assert B_q == B_k and d_model_q == d_model_k == self.d_model
        if attn_mask is not None:
            _B, _L_q, _L_k = attn_mask.size()
            assert _B == B_q and _L_q == L_q and _L_k == L_k

        n_h, residual = self.n_head, q
        assert d_model_q % n_h == 0, 'd_model {} divided by n_head {}.'.format(d_model_q, n_head)

        q_s = q.repeat(n_h, 1, 1).view(n_h, -1, d_model_q) # (n_head, B*L_q, d_model)
        k_s = k.repeat(n_h, 1, 1).view(n_h, -1, d_model_k) # (n_head, B*L_k, d_model)
        v_s = v.repeat(n_h, 1, 1).view(n_h, -1, d_model_v) # (n_head, B*L_v, d_model)

        # n_head as batch size, multiply
        q_s = tc.bmm(q_s, self.w_q).view(-1, L_q, self.d_k) # (B*n_head, L_q, d_k)
        k_s = tc.bmm(k_s, self.w_k).view(-1, L_k, self.d_k) # (B*n_head, L_k, d_k)
        v_s = tc.bmm(v_s, self.w_v).view(-1, L_v, self.d_v) # (B*n_head, L_v, d_v)

        # (B*n_head, trg_L, src_L)
        attn = tc.bmm(q_s, k_s.permute(0, 2, 1)) / self.temper  # (B*n_head, L_q, L_k)

        if attn_mask is not None:   # (B, L_q, L_k)
            attn_mask = attn_mask.repeat(n_h, 1, 1) # -> (n_head*B, L_q, L_k)
            assert attn_mask.size() == attn.size(), 'Attention mask shape {} mismatch ' \
                    'with Attention logit tensor shape {}.'.format(attn_mask.size(), attn.size())
            attn.data.masked_fill_(attn_mask, -float('inf'))
            #attn_mask = Variable(attn_mask.float(), requires_grad=False)

        attn = self.mSoftMax(attn)
        #print attn.cpu().data.numpy()
        #attn = self.mSoftMax(attn, mask=attn_mask, dim=-1)

        # one attention
        one_head_attn = attn.view(B_q, n_h, L_q, L_k)[:, 0, :, :].contiguous()

        attn = self.dropout(attn)   # (B*n_head, L_q, L_k)
        output = tc.bmm(attn, v_s)  # (B*n_head, L_q, d_v)  note: L_k == L_v
        # back to original batch size B
        #output = output.view(B_q, L_q, -1)  # (B_q, L_q, n_head*d_v)   can not use this !!!!
        output = tc.cat(tc.split(output, B_v, dim=0), dim=-1)
        output = self.proj(output)          # (B_q, L_q, d_model)

        return self.layer_norm(self.dropout(output) + residual), attn, one_head_attn

class PositionwiseFeedForward(nn.Module):
    '''
        A two-layer Feed-Forward Network
        Args:
            size(int): the size of input for the first-layer of the FFN.
            hidden_size(int): the hidden layer size of the second-layer
                              of the FNN.
            droput(float): dropout probability(0-1.0).
    '''
    def __init__(self, d_hid, d_inner_hid, dropout=0.1):

        super(PositionwiseFeedForward, self).__init__()
        #self.w_1 = InitLinear(size, hidden_size)
        #self.w_2 = InitLinear(hidden_size, size)
        self.w_1 = nn.Conv1d(d_hid, d_inner_hid, 1) # position-wise
        self.w_2 = nn.Conv1d(d_inner_hid, d_hid, 1) # position-wise
        #self.layer_norm = LayerNormalization(size)
        self.layer_norm = Layer_Norm(d_hid)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x    # (B_q, L_q, d_model)
        output = self.w_2(self.relu(self.w_1(x.permute(0, 2, 1)))).permute(0, 2, 1)
        return self.layer_norm(self.dropout(output) + residual) # dan

class EncoderLayer(nn.Module):
    '''
        Args:
            size(int): the dimension of keys/values/queries in
                       MultiHeadAttention, also the input size of
                       the first-layer of the PositionwiseFeedForward.
            droput(float): dropout probability(0-1.0).
            n_head(int): the number of head for MultiHeadAttention.
            hidden_size(int): the second-layer of the PositionwiseFeedForward.
    '''
    def __init__(self, d_model, n_head=8, d_k=64, d_v=64, d_inner_hid=2048, dropout=0.1):

        super(EncoderLayer, self).__init__()
        self.src_slf_attn = MultiHeadAttention(d_model, n_head, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner_hid, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        # q - k - v
        enc_output, enc_slf_attn, enc_slf_one_attn = self.src_slf_attn(
            enc_input, enc_input, enc_input, attn_mask=slf_attn_mask)
        # enc_output: (B_q, L_q, d_model), enc_slf_attn: (B*n_head, L_q, L_k)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn, enc_slf_one_attn

''' A encoder model with self attention mechanism. '''
class Encoder(nn.Module):

    def __init__(self, n_src_vocab, n_max_seq, n_layers=6, n_head=8, d_k=64, d_v=64,
            d_word_vec=512, d_model=512, d_inner_hid=1024, dropout=0.1):

        super(Encoder, self).__init__()

        n_position = n_max_seq + 1
        self.n_max_seq = n_max_seq
        self.d_model = d_model

        self.position_enc = nn.Embedding(n_position, d_word_vec, padding_idx=PAD)
        self.position_enc.weight.data = position_encoding_init(n_position, d_word_vec)

        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=PAD)
        wlog('src position emb: {}'.format(self.position_enc.weight.data.size()))
        wlog('src emb: {}'.format(self.src_word_emb.weight.data.size()))

        #print 'src: ', n_src_vocab, n_position
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, n_head, d_k, d_v, d_inner_hid, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, src_seq, src_pos):

        B, L = src_seq.size()
        # Word embedding look up
        enc_output = self.src_word_emb(src_seq)
        # Position Encoding addition
        enc_output += self.position_enc(src_pos)
        enc_outputs, enc_slf_attns = [], []

        #src_slf_attn_mask = src_seq.data.ne(PAD).unsqueeze(1).expand(B, L, L)
        src_slf_attn_mask = src_seq.data.eq(PAD).unsqueeze(1).expand(B, L, L)
        #src_slf_attn_mask = get_attn_padding_mask(src_seq, src_seq)
        for enc_layer in self.layer_stack:
            # enc_output: (B_q, L_q, d_model), enc_slf_attn: (B*n_head, L_q, L_k)
            enc_output, enc_slf_attn, enc_slf_one_attn = enc_layer(enc_output, src_slf_attn_mask)
            enc_outputs += [enc_output]
            enc_slf_attns += [enc_slf_attn]

        return (enc_outputs, enc_slf_attns, enc_slf_one_attn)

class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, n_head, d_k=64, d_v=64, d_inner_hid=2048, dropout=0.1):

        super(DecoderLayer, self).__init__()
        self.trg_slf_attn = MultiHeadAttention(d_model, n_head, d_k, d_v, dropout=dropout)
        self.trg_src_attn = MultiHeadAttention(d_model, n_head, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner_hid, dropout=dropout)

    def forward(self, dec_input, enc_output, trg_slf_attn_mask=None, trg_src_attn_mask=None):
        # trg_slf_attn_mask: (B, trg_L, trg_L), trg_src_attn_mask: (B, trg_L, src_L)
        dec_output, dec_slf_attn, dec_slf_one_attn = self.trg_slf_attn(
            dec_input, dec_input, dec_input, attn_mask=trg_slf_attn_mask)
        # (L_q, L_k, L_v) == (trg_L, trg_L, trg_L)
        # dec_output: (B_q, L_q, d_model) == (B, trg_L, d_model)
        # dec_slf_attn: (B*n_head, L_q, L_k) == (B*n_head, trg_L, trg_L)

        dec_output, dec_enc_attn, dec_enc_one_attn = self.trg_src_attn(
            dec_output, enc_output, enc_output, attn_mask=trg_src_attn_mask)
        # (L_q, L_k, L_v) == (trg_L, src_L, src_L)
        # dec_output: (B_q, L_q, d_model) == (B, trg_L, d_model)
        # dec_enc_attn: (B*n_head, L_q, L_k) == (B*n_head, trg_L, src_L)

        dec_output = self.pos_ffn(dec_output)

        return dec_output, dec_slf_attn, dec_enc_attn, dec_enc_one_attn

class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''
    def __init__(self, n_tgt_vocab, n_max_seq, n_layers=6, n_head=8, d_k=64, d_v=64,
            d_word_vec=512, d_model=512, d_inner_hid=1024, dropout=0.1, proj_share_weight=False):

        super(Decoder, self).__init__()
        n_position = n_max_seq + 1
        self.n_max_seq = n_max_seq
        self.d_model = d_model

        self.position_enc = nn.Embedding(n_position + 2, d_word_vec, padding_idx=PAD)
        self.position_enc.weight.data = position_encoding_init(n_position + 2, d_word_vec)

        self.tgt_word_emb = nn.Embedding(n_tgt_vocab, d_word_vec, padding_idx=PAD)
        wlog('trg position emb: {}'.format(self.position_enc.weight.data.size()))
        wlog('trg emb: {}'.format(self.tgt_word_emb.weight.data.size()))

        trg_lookup_table = self.tgt_word_emb if proj_share_weight is True else None
        self.classifier = Classifier(d_model, n_tgt_vocab, trg_lookup_table,
                                     trg_wemb_size=d_word_vec)

        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, n_head, d_k, d_v, d_inner_hid, dropout=dropout)
            for _ in range(n_layers)])

    #def forward(self, tgt_seq, tgt_pos, src_seq, enc_outputs):
    def forward(self, tgt_seq, tgt_pos, src_seq, enc_output):

        src_B, src_L = src_seq.size()
        trg_B, trg_L = tgt_seq.size()
        #print trg_L, tgt_pos.size(-1)
        assert src_B == trg_B
        # Word embedding look up
        dec_out = self.tgt_word_emb(tgt_seq)
        dec_out += self.position_enc(tgt_pos)

        '''
        Get an attention mask to avoid using the subsequent info.
        array([[[0, 1, 1],
                [0, 0, 1],
                [0, 0, 0]]], dtype=uint8)
        '''
        trg_src_attn_mask = src_seq.data.eq(PAD).unsqueeze(1).expand(src_B, trg_L, src_L)

        trg_slf_attn_mask = tgt_seq.data.eq(PAD).unsqueeze(1).expand(trg_B, trg_L, trg_L)
        subsequent_mask = np.triu(np.ones((trg_B, trg_L, trg_L)), k=1).astype('uint8')
        subsequent_mask = tc.from_numpy(subsequent_mask)
        if tgt_seq.is_cuda: subsequent_mask = subsequent_mask.cuda()
        trg_slf_attn_mask = tc.gt(trg_slf_attn_mask + subsequent_mask, 0)
        # Decode
        #dec_slf_attn_pad_mask = get_attn_padding_mask(tgt_seq, tgt_seq)
        #dec_slf_attn_sub_mask = get_attn_subsequent_mask(tgt_seq)
        #trg_slf_attn_mask = tc.gt(dec_slf_attn_pad_mask + dec_slf_attn_sub_mask, 0)

        #trg_src_attn_mask = get_attn_padding_mask(tgt_seq, src_seq)
        # (mb_size, len_q, len_k)  len_q == len_k == len_trg
        # (mb_size, len_q, len_k)  len_q == len_trg, len_k == len_src
        dec_slf_attns, dec_enc_attns = [], []

        #for dec_layer, enc_output in zip(self.layer_stack, enc_outputs):
        for dec_layer in self.layer_stack:
            dec_out, dec_slf_attn, dec_enc_attn, dec_enc_one_attn = dec_layer(dec_out, enc_output,
                trg_slf_attn_mask=trg_slf_attn_mask, trg_src_attn_mask=trg_src_attn_mask)
            dec_slf_attns += [dec_slf_attn]
            dec_enc_attns += [dec_enc_attn]

        return (dec_out, dec_slf_attns, dec_enc_attns, dec_enc_one_attn)

def position_encoding_init(n_position, d_pos_vec):
    ''' Init the sinusoid position encoding table '''

    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [pos / np.power(10000, 2*i/d_pos_vec) for i in range(d_pos_vec)]
        if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1
    return tc.from_numpy(position_enc).type(tc.FloatTensor)


