import math
import torch as tc
import torch.nn as nn

epsilon = 1e-20

class MaskSoftmax(nn.Module):

    def __init__(self):

        super(MaskSoftmax, self).__init__()

    def forward(self, x, mask=None, dim=-1):

        # input torch tensor or variable, take max for numerical stability
        x_max = tc.max(x, dim=dim, keepdim=True)[0]
        x_minus = x - x_max
        x_exp = tc.exp(x_minus)
        if mask is not None: x_exp = x_exp * mask
        x = x_exp / ( tc.sum( x_exp, dim=dim, keepdim=True ) + epsilon )

        return x

class MyLogSoftmax(nn.Module):

    def __init__(self, self_norm_alpha=None):

        super(MyLogSoftmax, self).__init__()
        self.sna = self_norm_alpha

    def forward(self, x, dim=-1):

        # input torch tensor
        x_max = tc.max(x, dim=dim, keepdim=True)[0]  # take max for numerical stability
        x_exp = tc.exp( x - x_max )
        x_exp_sum = tc.sum( x_exp, dim=dim, keepdim=True ) + epsilon
        log_norm = tc.log( x_exp_sum ) + x_max
        x = x - log_norm    # get log softmax
        prob = x_exp / x_exp_sum

        # Sum_( log(P(xi)) - alpha * square( log(Z(xi)) ) )
        if self.sna is not None: x = x - self.sna * tc.pow(log_norm, 2)

        return log_norm, prob, x

'''Layer normalize the tensor x, averaging over the last dimension.'''
class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(tc.ones(features))
        self.b_2 = nn.Parameter(tc.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

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

class PositionwiseFeedForward(nn.Module):
    '''
        A two-layer Feed-Forward Network
        Args:
            size(int): the size of input for the first-layer of the FFN.
            hidden_size(int): the hidden layer size of the second-layer of the FNN.
            droput(float): dropout probability(0-1.0).
    '''
    def __init__(self, input_size=512, filter_size=2048, output_size=512, dropout_prob=0.1):

        super(PositionwiseFeedForward, self).__init__()
        self.filter_transform = nn.Linear(input_size, filter_size, bias=True)
        self.relu = nn.ReLU()
        if dropout_prob is not None and 0. < dropout_prob <= 1.0:
            self.dropout = nn.Dropout(dropout_prob)
        self.dropout_prob = dropout_prob
        self.output_transform = nn.Linear(filter_size, output_size, bias=True)

    def forward(self, x):

        # (batch_size, input_len, model_dim) -> (batch_size, input_len, model_dim)
        hidden = self.filter_transform(x)
        hidden = self.relu(hidden)

        if self.dropout_prob is not None and 0. < self.dropout_prob <= 1.0:
            hidden = self.dropout(hidden)

        output = self.output_transform(hidden)

        return output




