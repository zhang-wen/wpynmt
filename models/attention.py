import math
import torch as tc
import torch.nn as nn
from nn_utils import MaskSoftmax

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

    #d_model(int):   the dimension of n_head keys/values/queries: d_model % n_head == 0
    #n_head(int):    number of parallel heads.
    def __init__(self, enc_hid_size, d_model, n_head=8):

        super(Multihead_Additive_Attention, self).__init__()

        assert d_model % n_head == 0, 'd_model {} divided by n_head {}.'.format(d_model, n_head)
        dim_per_head = d_model // n_head
        self.n_head = n_head

        self.linear_query = nn.Linear(d_model, n_head * dim_per_head, bias=False)
        self.mSoftMax = MaskSoftmax()
        self.tanh = nn.Tanh()
        self.a1 = nn.Linear(dim_per_head, 1, bias=False)
        self.final_proj = nn.Linear(2 * d_model, 2 * d_model, bias=True)

    '''
        Compute the context vector and the attention vectors.
        Args:
           k (FloatTensor): key vectors [batch_size, key_len, d_model]      ->  uh
           v (FloatTensor): value vectors [batch_size, key_len, 2*d_model]  ->  annotations
           q (FloatTensor): query vectors  [batch_size, d_model]            ->  hidden state
           attn_mask: binary mask indicating
                    which keys have non-zero attention [batch_size, key_len]
        Returns:
           (FloatTensor, FloatTensor) :
           * output context vectors [batch_size, 2 * d_model]
           * probability            [batch_size, n_head, key_len]
    '''
    def forward(self, q, v, k, attn_mask=None):

        q = q[:, None, :]
        hidden_size = q.size(-1)
        batch_size, key_len = k.size(0), k.size(1)

        def reshape_head(x, nhead):
            return x.view(x.size(0), x.size(1), nhead, x.size(-1) // nhead).permute(0, 2, 1, 3)

        def unshape_head(x, nhead):
            return x.permute(0, 2, 1, 3).contiguous().view(x.size(0), x.size(2), nhead * x.size(-1))

        # 1. project key, value, and query
        key_up = reshape_head(k, self.n_head)                     # [batch_size, n_head, key_len, dim_per_head]
        value_up = reshape_head(v, self.n_head)                   # [batch_size, n_head, key_len, 2*dim_per_head]
        query_up = reshape_head(self.linear_query(q), self.n_head)# [batch_size, n_head, 1, dim_per_head]

        hidden = self.tanh(query_up.expand_as(key_up) + key_up)

        attn = self.a1(hidden).squeeze(-1)  # [batch_size, n_head, key_len]

        if attn_mask is not None:   # [batch_size, key_len]
            attn_mask = attn_mask.unsqueeze(1).expand_as(attn).byte()    # expand along n_head dim
            assert attn_mask.size() == attn.size(), 'Attention mask shape {} mismatch ' \
                    'with Attention logit tensor shape {}.'.format(attn_mask.size(), attn.size())
            attn = attn.masked_fill_(1 - attn_mask, -1e18)

        # 3. apply attention dropout and compute context vectors
        alpha = self.mSoftMax(attn)                 # [batch_size, n_head, key_len]
        attn = alpha[:, :, :, None] * value_up      # [batch_size, n_head, key_len, 2*dim_per_head]
        attn = unshape_head(attn, self.n_head)      # [batch_size, key_len, 2*d_model]

        attn = self.final_proj(attn.sum(1))       # [batch_size, 2 * d_model]

        alpha = alpha[:, 0, :]  # get the attention of the first head, [batch_size, key_len]
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
        self.dropout = nn.Dropout(dropout_prob)
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

        def reshape_head(x):
            return x.view(batch_size, -1, n_head, dim_per_head).transpose(1, 2)

        def unshape_head(x):
            return x.transpose(1, 2).contiguous().view(batch_size, -1, n_head * dim_per_head)

        # 1. project key, value, and query
        key_up = reshape_head(self.linear_keys(k)) # [batch_size, n_head, key_len, dim_per_head]
        value_up = reshape_head(self.linear_values(v)) # [batch_size, n_head, key_len, dim_per_head]
        query_up = reshape_head(self.linear_query(q))  # [batch_size, n_head, query_len, dim_per_head]

        # 2. calculate and scale scores: Attention(Q,K,V) = softmax(QK/sqrt(d_k))*V
        query_up = query_up / math.sqrt(dim_per_head)# [batch_size, n_head, query_len, dim_per_head]
        attn = tc.matmul(query_up, key_up.transpose(2, 3))#[batch_size, n_head, query_len, key_len]

        if attn_mask is not None:   # [batch_size, query_len, key_len]
            attn_mask = attn_mask.unsqueeze(1).expand_as(attn).byte()    # expand along n_head dim
            assert attn_mask.size() == attn.size(), 'Attention mask shape {} mismatch ' \
                    'with Attention logit tensor shape {}.'.format(attn_mask.size(), attn.size())
            attn.masked_fill_(attn_mask, -1e18)

        # 3. apply attention dropout and compute context vectors
        attn = self.mSoftMax(attn)
        attn_drop = self.dropout(attn)              # [batch_size, n_head, query_len, key_len]
        context = tc.matmul(attn_drop, value_up)    # [batch_size, n_head, query_len, dim_per_head]
        context = unshape_head(context)             # [batch_size, query_len, n_head * dim_per_head]

        output = self.final_proj(context)   # [batch_size, query_len, d_model]

        return output, attn



