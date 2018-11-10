import numpy as np
import torch as tc
import torch.nn as nn
from tools.utils import MAX_SEQ_SIZE, wlog, PAD
from nn_utils import PositionwiseFeedForward, MultiHeadAttention

'''
Get an attention mask to avoid using the subsequent info.
Args: d_model: int
Returns: (LongTensor): subsequent_mask [1, d_model, d_model]
'''
def get_attn_subsequent_mask(size):

    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    subsequent_mask = tc.from_numpy(subsequent_mask)

    return subsequent_mask

'''
Compose with three layers
    Args:
        d_model(int): the dimension of keys/values/queries in
                      MultiHeadAttention, also the input size of
                      the first-layer of the PositionwiseFeedForward.
        n_head(int): the number of head for MultiHeadAttention.
        hidden_size(int): the second-layer of the PositionwiseFeedForward.
        droput(float): dropout probability(0-1.0).
'''
class SelfAttDecoderLayer(nn.Module):

    def __init__(self,
                 d_model,
                 n_head=8,
                 d_ff_filter=2048,
                 att_dropout=0.3,
                 residual_dropout=0.,
                 relu_dropout=0.,
                 self_attn_type='scaled-dot'):

        super(SelfAttDecoderLayer, self).__init__()

        self.layer_norm_0 = nn.LayerNorm(d_model, eps=1e-6, elementwise_affine=True)

        self.self_attn_type = self_attn_type
        if self_attn_type == 'scaled-dot':
            self.self_attn = MultiHeadAttention(d_model, n_head, dropout_prob=att_dropout)
        elif self_attn_type == 'average':
            self.self_attn = AverageAttention(d_model, dropout_prob=att_dropout)

        self.drop_residual_0= nn.Dropout(residual_dropout)

        self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-6, elementwise_affine=True)
        self.trg_src_attn = MultiHeadAttention(d_model, n_head, dropout_prob=att_dropout)
        self.drop_residual_1 = nn.Dropout(residual_dropout)

        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6, elementwise_affine=True)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_ff_filter, d_model, dropout_prob=relu_dropout)
        self.drop_residual_2 = nn.Dropout(residual_dropout)

        subsequent_mask = get_attn_subsequent_mask(MAX_SEQ_SIZE)
        # Register self.mask as a buffer in TransformerDecoderLayer, so
        # it gets TransformerDecoderLayer's cuda behavior automatically.
        self.register_buffer('subsequent_mask', subsequent_mask)

    def forward(self, dec_inputs, enc_output, trg_self_attn_mask=None, trg_src_attn_mask=None):
        '''
        Args:
            dec_inputs (FloatTensor):       [batch_size, trg_len, d_model]
            enc_output (FloatTensor):       [batch_size, src_len, d_model]
            trg_self_attn_mask (LongTensor):[batch_size, trg_len, trg_len]
            trg_src_attn_mask  (LongTensor):[batch_size, trg_len, src_len]
        Returns: (FloatTensor, FloatTensor, FloatTensor, FloatTensor):
            dec_output:         [batch_size, trg_len, d_model]
            dec_self_attns:     [batch_size, n_head, trg_len, trg_len]
            dec_enc_attns:      [batch_size, n_head, trg_len, src_len]
            one_dec_enc_attn:   [batch_size, trg_len, src_len]
        '''

        trg_len = trg_self_attn_mask.size(1)
        dec_mask = tc.gt(trg_self_attn_mask + self.subsequent_mask[:, :trg_len, :trg_len], 0)

        # target self-attention
        norm_inputs = self.layer_norm_0(dec_inputs)     # 'n' for preprocess

        # dec_mask: (batch_size, trg_len, trg_len)
        if self.self_attn_type == 'scaled-dot':
            query, dec_self_attns = self.self_attn(
                norm_inputs, norm_inputs, norm_inputs, attn_mask=dec_mask)
            # query:                [batch_size, trg_len, d_model]
            # dec_self_attns:       [batch_size, n_head, trg_len, trg_len]
            # one_dec_self_attn:    [batch_size, trg_len, trg_len]
        elif self.self_attn_type == 'average':
            query, attn = self.self_attn(input_norm, mask=dec_mask,
                                         layer_cache=layer_cache, step=step)

        query = self.drop_residual_0(query) + dec_inputs  # 'da' for postprocess

        # encoder-decoder attention
        norm_query = self.layer_norm_1(query)   # 'n' for preprocess

        # trg_src_attn_mask: (batch_size, trg_en, src_len)
        dec_output, dec_enc_attns = self.trg_src_attn(
            enc_output, enc_output, norm_query, attn_mask=trg_src_attn_mask)
        # dec_output:           [batch_size, trg_len, d_model]
        # dec_enc_attns:        [batch_size, n_head, trg_len, src_len]
        # one_dec_enc_attn:     [batch_size, trg_len, src_len]

        x = self.drop_residual_1(dec_output) + query    # 'da' for postprocess

        # feed forward
        norm_x = self.layer_norm_2(x)   # 'n' for preprocess

        dec_output = self.pos_ffn(norm_x)

        dec_output = self.drop_residual_2(dec_output) + x     # 'da' for postprocess

        return dec_output, dec_self_attns, dec_enc_attns

''' A decoder model with self attention mechanism '''
class SelfAttDecoder(nn.Module):

    def __init__(self, trg_emb,
                 n_layers=6,
                 d_model=512,
                 n_head=8,
                 d_ff_filter=1024,
                 att_dropout=0.3,
                 residual_dropout=0.,
                 relu_dropout=0.,
                 self_attn_type='scaled-dot',
                 proj_share_weight=False):

        wlog('Transformer decoder ========================= ')
        wlog('\ttrg_word_emb:       {}'.format(trg_emb.we.weight.size()))
        wlog('\tn_layers:           {}'.format(n_layers))
        wlog('\tn_head:             {}'.format(n_head))
        wlog('\td_word_vec:         {}'.format(trg_emb.we.weight.size(-1)))
        wlog('\td_model:            {}'.format(d_model))
        wlog('\td_ffn_filter:       {}'.format(d_ff_filter))
        wlog('\tatt_dropout:        {}'.format(att_dropout))
        wlog('\tresidual_dropout:   {}'.format(residual_dropout))
        wlog('\trelu_dropout:       {}'.format(relu_dropout))
        wlog('\tproj_share_weight:  {}'.format(proj_share_weight))

        super(SelfAttDecoder, self).__init__()

        self.embed = trg_emb

        self.layer_stack = nn.ModuleList([
            SelfAttDecoderLayer(d_model,
                                n_head,
                                d_ff_filter,
                                att_dropout=att_dropout,
                                residual_dropout=residual_dropout,
                                relu_dropout=relu_dropout,
                                self_attn_type=self_attn_type)
            for _ in range(n_layers)])

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6, elementwise_affine=True)

    def forward(self, trg_seq, src_seq, enc_output):

        src_B, src_L = src_seq.size()
        trg_B, trg_L = trg_seq.size()
        assert src_B == trg_B

        '''
        Get an attention mask to avoid using the subsequent info.
        array([[[0, 1, 1],
                [0, 0, 1],
                [0, 0, 0]]], dtype=uint8)
        '''
        trg_src_attn_mask = src_seq.data.eq(PAD).unsqueeze(1).expand(src_B, trg_L, src_L)
        trg_self_attn_mask = trg_seq.data.eq(PAD).unsqueeze(1).expand(trg_B, trg_L, trg_L)

        dec_output = self.embed(trg_seq)

        nlayer_outputs, nlayer_self_attns, nlayer_attns = [], [], []
        for dec_layer in self.layer_stack:
            dec_output, dec_self_attns, dec_enc_attns = dec_layer(
                dec_output, enc_output,
                trg_self_attn_mask=trg_self_attn_mask,
                trg_src_attn_mask=trg_src_attn_mask)
            #nlayer_outputs += [dec_output]
            nlayer_self_attns += [dec_self_attns]
            nlayer_attns += [dec_enc_attns]

        dec_output = self.layer_norm(dec_output)    # layer norm for the last layer output

        return (dec_output, nlayer_self_attns, nlayer_attns)


