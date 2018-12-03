from __future__ import print_function

import math
import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from nn_utils import MaskSoftmax
import numpy as np
np.set_printoptions(threshold='nan')

class Additive_Attention(nn.Module):

    def __init__(self, dec_hid_size, align_size):

        super(Additive_Attention, self).__init__()
        self.sa = nn.Linear(dec_hid_size, align_size)
        self.tanh = nn.Tanh()
        self.maskSoftmax = MaskSoftmax()
        self.a1 = nn.Linear(align_size, 1)

    def forward(self, s_tm1, xs_h, uh, xs_mask=None):

        e_ij = self.a1( self.tanh(self.sa(s_tm1)[:, None, :] + uh) ).squeeze(-1)

        e_ij = self.maskSoftmax(e_ij, mask=xs_mask, dim=1)  # (batch_size, key_len)
        # weighted sum of the h_j: (batch_size, enc_hid_size)
        attend = (e_ij[:, :, None] * xs_h).sum(1)

        return e_ij, attend

class Multihead_Additive_Attention(nn.Module):

    #dec_hid_size:   the dimension of n_head keys/values/queries: dec_hid_size % n_head == 0
    #n_head:    number of parallel heads.
    def __init__(self, enc_hid_size, dec_hid_size, n_head=8):

        super(Multihead_Additive_Attention, self).__init__()

        assert dec_hid_size % n_head == 0, 'dec_hid_size {} divided by n_head {}.'.format(dec_hid_size, n_head)
        self.n_head = n_head
        self.linear_query = nn.Linear(dec_hid_size, dec_hid_size, bias=False)
        #self.mSoftMax = MaskSoftmax()
        dim_per_head = dec_hid_size // n_head
        self.a1 = nn.Linear(dim_per_head, 1, bias=False)
        self.final_proj = nn.Linear(2 * dec_hid_size, 2 * dec_hid_size, bias=True)

    '''
        Compute the context vector and the attention vectors.
        Args:
           q (FloatTensor): query [batch_size, dec_hid_size]             ->  hidden state
           v (FloatTensor): value [batch_size, key_len, 2*dec_hid_size]  ->  annotations
           k (FloatTensor): key [batch_size, key_len, dec_hid_size]      ->  uh
           attn_mask: binary mask indicating
                    which keys have non-zero attention [batch_size, key_len]
        Returns:
           (FloatTensor, FloatTensor) :
           * output context vectors [batch_size, 2 * dec_hid_size]
           * probability            [batch_size, n_head, key_len]
    '''
    def forward(self, q, v, k, attn_mask=None):

        def split_heads(x, nhead):
            return x.view(x.size(0), x.size(1), nhead, x.size(-1) // nhead).permute(0, 2, 1, 3)

        def combine_heads(x, nhead):
            return x.permute(0, 2, 1, 3).contiguous().view(x.size(0), x.size(2), nhead * x.size(-1))

        q = self.linear_query(q)
        # 1. project key, value, and query
        q = split_heads(q[:, None, :], self.n_head) # [batch_size, n_head, 1, dim_per_head]
        k = split_heads(k, self.n_head)             # [batch_size, n_head, key_len, dim_per_head]

        hidden = tc.tanh(q + k)
        attn = self.a1(hidden).squeeze(-1)          # [batch_size, n_head, key_len]
        if attn_mask is not None:   # [batch_size, key_len]
            attn_mask = attn_mask.unsqueeze(1).expand_as(attn).byte()    # expand along n_head dim
            assert attn_mask.size() == attn.size(), 'Attention mask shape {} mismatch ' \
                    'with Attention logit tensor shape {}.'.format(attn_mask.size(), attn.size())
            attn = attn.masked_fill_(1 - attn_mask, float('-inf'))

        # 3. apply attention dropout and compute context vectors
        #alpha = self.mSoftMax(attn)            # [batch_size, n_head, key_len]
        alpha = F.softmax(attn, dim=-1)         # [batch_size, n_head, key_len]

        v = split_heads(v, self.n_head)             # [batch_size, n_head, key_len, 2*dim_per_head]
        attn = alpha[:, :, :, None] * v             # [batch_size, n_head, key_len, 2*dim_per_head]
        attn = combine_heads(attn, self.n_head)     # [batch_size, key_len, 2*d_model]

        attn = self.final_proj(attn.sum(1))       # [batch_size, 2 * d_model]

        alpha = alpha.sum(1) / self.n_head # get the attention of the first head, [batch_size, key_len]
        #alpha = alpha[:, 0, :].transpose(0, 1)  # get the attention of the first head, [key_len, batch_size]

        return alpha, attn

class MultiHeadAttention(nn.Module):

    #d_model(int):   the dimension of n_head keys/values/queries: d_model % n_head == 0
    #n_head(int):    number of parallel heads.
    def __init__(self, d_model, n_head, dropout_prob=0.1):

        super(MultiHeadAttention, self).__init__()

        assert d_model % n_head == 0, 'd_model {} divided by n_head {}.'.format(d_model, n_head)
        self.dim_per_head = d_model // n_head
        self.d_model = d_model
        self.n_head = n_head

        self.linear_keys = nn.Linear(d_model, n_head * self.dim_per_head)
        self.linear_values = nn.Linear(d_model, n_head * self.dim_per_head)
        self.linear_query = nn.Linear(d_model, n_head * self.dim_per_head)
        self.mSoftMax = MaskSoftmax()
        self.dropout_prob = dropout_prob
        self.final_proj = nn.Linear(d_model, d_model)

    '''
        Compute the context vector and the attention vectors.
        Args:
           k (FloatTensor): key vectors [batch_size, key_len, d_model]
           v (FloatTensor): value vectors [batch_size, key_len, d_model]
           q (FloatTensor): query vectors  [batch_size, query_len, d_model]
           attn_mask: binary mask indicating
                    which keys have non-zero attention [batch_size, query_len, key_len]
        Returns:
           (FloatTensor, FloatTensor, FloatTensor) :
           * output context vectors [batch_size, query_len, d_model]
           * all attention vectors [batch_size, n_head, query_len, key_len]
           * one of the attention vectors [batch_size, query_len, key_len]
    '''
    def forward(self, k, v, q, attn_mask=None):

        batch_size, key_len = k.size(0), k.size(1)
        dim_per_head = self.dim_per_head
        n_head = self.n_head
        query_len = q.size(1)

        def split_heads(x):
            return x.view(batch_size, -1, n_head, dim_per_head).transpose(1, 2)

        def combine_heads(x):
            return x.transpose(1, 2).contiguous().view(batch_size, -1, n_head * dim_per_head)

        # 1. project key, value, and query
        key_up = split_heads(self.linear_keys(k)) # [batch_size, n_head, key_len, dim_per_head]
        value_up = split_heads(self.linear_values(v)) # [batch_size, n_head, key_len, dim_per_head]
        query_up = split_heads(self.linear_query(q))  # [batch_size, n_head, query_len, dim_per_head]

        # 2. calculate and scale scores: Attention(Q,K,V) = softmax(QK/sqrt(d_k))*V
        query_up = query_up / math.sqrt(dim_per_head)# [batch_size, n_head, query_len, dim_per_head]
        attn = tc.matmul(query_up, key_up.transpose(2, 3))#[batch_size, n_head, query_len, key_len]

        if attn_mask is not None:   # [batch_size, query_len, key_len]
            attn_mask = attn_mask.unsqueeze(1).expand_as(attn).byte()    # expand along n_head dim
            assert attn_mask.size() == attn.size(), 'Attention mask shape {} mismatch ' \
                    'with Attention logit tensor shape {}.'.format(attn_mask.size(), attn.size())
            attn.masked_fill_(attn_mask, float('-inf'))

        # 3. apply attention dropout and compute context vectors
        attn = self.mSoftMax(attn, dim=-1)
        #attn = F.softmax(attn, dim=-1)
        attn_weights = F.dropout(attn, p=self.dropout_prob, training=self.training) # [batch_size, n_head, query_len, key_len]
        #context = tc.matmul(attn, value_up)    # [batch_size, n_head, query_len, dim_per_head]
        context = tc.bmm(attn.contiguous().view(-1, query_len, key_len),
                         value_up.contiguous().view(-1, key_len, dim_per_head))    # [batch_size, n_head, query_len, dim_per_head]
        context = context.view(batch_size, n_head, query_len, dim_per_head)
        context = combine_heads(context)             # [batch_size, query_len, n_head * dim_per_head]

        output = self.final_proj(context)   # [batch_size, query_len, d_model]

        attn = attn.sum(dim=1) / self.n_head    # average attention weights over heads

        return output, attn



