import math
import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from tools.utils import PAD, MAX_SEQ_SIZE, wlog
import wargs

'''
Implements the sinusoidal positional encoding for non-recurrent neural networks
Args:
   dropout (float): dropout parameter
   n_embed (int): embedding size
'''
class PositionalEncoding(nn.Module):

    def __init__(self, dropout_prob, n_embed, max_len=MAX_SEQ_SIZE):

        pe = tc.zeros(max_len, n_embed)
        position = tc.arange(0, max_len).unsqueeze(1)
        div_term = tc.exp((tc.arange(0, n_embed, 2) * -(math.log(10000.0) / n_embed)).float())
        inter_term = position.float() * div_term
        # keep dim 0 for padding token position encoding zero vector
        pe[1:, 0::2] = tc.sin(inter_term)[1:]
        pe[1:, 1::2] = tc.cos(inter_term)[1:]
        # [5000, 1] * [256] = [5000, 256] 
        #pe[:, 0::2] = tc.sin(position.float() * div_term)
        #pe[:, 1::2] = tc.cos(position.float() * div_term)
        pe = pe.unsqueeze(1)    # [5000, 512] -> [5000, 1, 512]
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.n_embed = n_embed
        wlog('pe: {}'.format(pe.size()))

        self.dropout_prob =  dropout_prob
        if dropout_prob is not None and 0. < dropout_prob <= 1.0:
            wlog('with emb dropout prob = {} ...'.format(dropout_prob))
            self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, emb):

        emb = emb * math.sqrt(self.n_embed)
        emb = emb + self.pe[:emb.size(0)]
        if self.dropout_prob is not None and 0. < self.dropout_prob < 1.0: emb = self.dropout(emb)

        return emb

class WordEmbedding(nn.Module):

    def __init__(self,
                 n_vocab,
                 n_embed=512,
                 emb_dropout=None,
                 position_encoding=False,
                 prefix='WordEmbedding'):

        super(WordEmbedding, self).__init__()
        wlog('WordEmbedding_{}'.format(prefix))
        self.position_encoding = position_encoding
        self.we = nn.Embedding(n_vocab, n_embed, padding_idx=PAD)
        self.n_embed = n_embed
        if position_encoding is True:
            wlog('with position emb ...')
            #self.pe = PositionalEncoding(emb_dropout, n_embed)
        if emb_dropout is not None and 0. < emb_dropout <= 1.0:
            wlog('with emb dropout prob = {} ...'.format(emb_dropout))
            self.dropout = nn.Dropout(p=emb_dropout)
        self.emb_dropout = emb_dropout

    def add_timing_signal(self, x_emb, min_timescale=1.0, max_timescale=1.0e4, name=None):

        length, channels = x_emb.size(1), x_emb.size(2)
        position = tc.arange(length).float()
        num_timescales = channels // 2

        log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) /
            (float(num_timescales) - 1)
        )
        inv_timescales = min_timescale * tc.exp(
            (tc.arange(num_timescales).float()) * -log_timescale_increment
        )

        scaled_time = position[:, None] * inv_timescales[None, :]
        signal = tc.cat([tc.sin(scaled_time), tc.cos(scaled_time)], dim=1)
        signal = F.pad(signal, (0, channels % 2, 0, 0))
        signal = signal.reshape(1, length, channels)
        if wargs.gpu_id is not None: signal = signal.cuda()

        return x_emb + signal * (float(channels) ** -0.5)

    def forward(self, x):

        x_w_emb = self.we(x)
        if self.position_encoding is True:
            x_wp_emb = self.add_timing_signal(x_w_emb)

        if self.emb_dropout is not None and 0. < self.emb_dropout <= 1.0:
            x_wp_emb = self.dropout(x_wp_emb)

        return x_w_emb, x_wp_emb






















