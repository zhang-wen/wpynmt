import torch.nn as nn
from tools.utils import *
from nn_utils import PositionwiseFeedForward, MultiHeadAttention

'''
    Args:
        d_model(int): the dimension of keys/values/queries in
                      MultiHeadAttention, also the input size of
                      the first-layer of the PositionwiseFeedForward.
        n_head(int): the number of head for MultiHeadAttention.
        hidden_size(int): the second-layer of the PositionwiseFeedForward.
        droput(float): dropout probability(0-1.0).
'''
class SelfAttEncoderLayer(nn.Module):

    def __init__(self,
                 d_model,
                 n_head=8,
                 d_ff_filter=2048,
                 att_dropout=0.3,
                 residual_dropout=0.0,
                 relu_dropout=0.0):

        super(SelfAttEncoderLayer, self).__init__()
        self.layer_norm_0 = nn.LayerNorm(d_model, eps=1e-6, elementwise_affine=True)
        self.self_attn = MultiHeadAttention(d_model, n_head, dropout_prob=att_dropout)
        self.dropout_0 = nn.Dropout(residual_dropout)
        self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-6, elementwise_affine=True)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_ff_filter, d_model, dropout_prob=relu_dropout)
        self.dropout_1 = nn.Dropout(residual_dropout)

    def forward(self, inputs, self_attn_mask=None):
        # inputs (FloatTensor):         [batch_size, src_L, d_model]
        # self_attn_mask(LongTensor):   [batch_size, src_L, src_L]
        # return:                       [batch_size, src_L, d_model]

        # self attention
        norm_inputs = self.layer_norm_0(inputs)   # 'n' for source self attention preprocess
        x, enc_self_attns = self.self_attn(
            norm_inputs, norm_inputs, norm_inputs, attn_mask=self_attn_mask)
        # enc_output: (B_q, L_q, d_model), enc_self_attns: (B_q, n_head, L_q, L_k)

        x = self.dropout_0(x) + inputs     # 'da' for self attention postprocess

        # feed forward
        norm_x = self.layer_norm_1(x)        # 'n' for feedforward preprocess
        enc_output = self.pos_ffn(norm_x)

        enc_output = self.dropout_1(enc_output) + x   # 'da' for feedforward postprocess

        # enc_output:           [batch_size, src_L, d_model]
        # enc_self_attns:       [batch_size, n_head, src_L, src_L]
        # one_enc_self_attn:    [batch_size, src_L, src_L]
        return enc_output, enc_self_attns

''' A encoder model with self attention mechanism '''
class SelfAttEncoder(nn.Module):

    def __init__(self,
                 src_emb,
                 n_layers=6,
                 d_model=512,
                 n_head=8,
                 d_ff_filter=2048,
                 att_dropout=0.3,
                 residual_dropout=0.0,
                 relu_dropout=0.0):

        super(SelfAttEncoder, self).__init__()

        wlog('Transformer encoder ========================= ')
        wlog('\tsrc_word_emb:       {}'.format(src_emb.we.weight.size()))
        wlog('\tn_layers:           {}'.format(n_layers))
        wlog('\tn_head:             {}'.format(n_head))
        wlog('\td_word_vec:         {}'.format(src_emb.we.weight.size(-1)))
        wlog('\td_model:            {}'.format(d_model))
        wlog('\td_ffn_filter:       {}'.format(d_ff_filter))
        wlog('\tatt_dropout:        {}'.format(att_dropout))
        wlog('\tresidual_dropout:   {}'.format(residual_dropout))
        wlog('\trelu_dropout:       {}'.format(relu_dropout))

        self.embed = src_emb

        self.layer_stack = nn.ModuleList([
            SelfAttEncoderLayer(d_model,
                                n_head,
                                d_ff_filter,
                                att_dropout=att_dropout,
                                residual_dropout=residual_dropout,
                                relu_dropout=relu_dropout)
            for _ in range(n_layers)])

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6, elementwise_affine=True)

    def forward(self, src_seq):

        batch_size, src_L = src_seq.size()
        # word embedding look up
        enc_output = self.embed(src_seq)
        nlayer_outputs, nlayer_attns = [], []
        src_self_attn_mask = src_seq.data.eq(PAD).unsqueeze(1).expand(batch_size, src_L, src_L) # 0. is 1 !!!
        for enc_layer in self.layer_stack:
            # enc_output: (B_q, L_q, d_model), enc_self_attns: (B, n_head, L_q, L_k)
            enc_output, enc_self_attns = enc_layer(enc_output, src_self_attn_mask)
            #nlayer_outputs += [enc_output]
            nlayer_attns += [enc_self_attns]

        enc_output = self.layer_norm(enc_output)    # layer norm for the last layer output

        # nlayer_outputs:   n_layers: [ [batch_size, src_L, d_model], ... ]
        # nlayer_attns:     n_layers: [ [batch_size, n_head, src_L, src_L], ... ]
        # one_enc_self_attn:          [batch_size, src_L, src_L]
        return (enc_output, nlayer_attns)


