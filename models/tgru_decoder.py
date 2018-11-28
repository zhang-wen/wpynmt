import torch as tc
import torch.nn as nn
import wargs
from tools.utils import *
from gru import LGRU, TGRU
from attention import Multihead_Additive_Attention

'''
    Transition Gated Recurrent Unit network decoder
    input args:
        trg_emb:        class WordEmbedding
        enc_hid_size:   the size of TGRU hidden state in encoder
        dec_hid_size:   the size of TGRU hidden state in decoder
        n_layers:       layer nubmer of decoder
'''
class StackedTransDecoder(nn.Module):

    def __init__(self, trg_emb, enc_hid_size=512, dec_hid_size=512, n_head=8, n_layers=3,
                 attention_type='multihead_additive', max_out=False,
                 rnn_dropout=0.3, out_dropout_prob=0.5,
                 prefix='TGRU_Decoder', **kwargs):

        super(StackedTransDecoder, self).__init__()

        self.s_init = nn.Linear(2 * enc_hid_size, dec_hid_size, bias=True)
        self.tanh = nn.Tanh()
        self.word_emb = trg_emb
        n_embed = trg_emb.n_embed
        f = lambda name: str_cat(prefix, name)  # return 'Encoder_' + parameters name

        self.lgru = LGRU(n_embed, dec_hid_size, dropout_prob=rnn_dropout, prefix=f('{}_{}'.format(prefix, 0)))
        self.tgrus = nn.ModuleList( [ TGRU(dec_hid_size, dropout_prob=rnn_dropout,
                                           prefix=f('{}_{}'.format(prefix, n)))
                                     for n in range(1, n_layers) ] )

        self.cond_lgru = LGRU(2 * enc_hid_size, dec_hid_size, dropout_prob=rnn_dropout,
                              prefix=f('{}_{}_{}'.format(prefix, 'cond', 0)))
        self.cond_tgrus = nn.ModuleList( [ TGRU(dec_hid_size, dropout_prob=rnn_dropout,
                                                prefix=f('{}_{}_{}'.format(prefix, 'cond', n)))
                                          for n in range(n_layers - 1) ] )

        self.n_layers = n_layers
        if attention_type == 'additive':
            self.keys_transform = nn.Linear(enc_hid_size, dec_hid_size)
            self.attention = Additive_Attention(dec_hid_size, wargs.align_size)
        if attention_type == 'multihead_additive':
            self.keys_transform = nn.Linear(2 * enc_hid_size, dec_hid_size, bias=False)
            self.attention = Multihead_Additive_Attention(enc_hid_size, dec_hid_size, n_head=n_head)

        self.sigmoid = nn.Sigmoid()
        self.s_transform = nn.Linear(dec_hid_size, dec_hid_size)
        self.y_transform = nn.Linear(n_embed, dec_hid_size)
        self.c_transform = nn.Linear(2*dec_hid_size, dec_hid_size)
        self.max_out = max_out
        if out_dropout_prob is not None and 0. < out_dropout_prob <= 1.0:
            self.output_dropout = nn.Dropout(p=out_dropout_prob)
        self.out_dropout_prob = out_dropout_prob

    def init_state(self, annotations, xs_mask=None):

        assert annotations.dim() == 3  # (batch_size, max_L, dec_hid_size)
        uh = self.keys_transform(annotations)
        if xs_mask is not None:
            annotations = (annotations * xs_mask[:, :, None]).sum(1) / xs_mask.sum(1)[:, None]
        else:
            annotations = annotations.mean(1)

        return self.tanh(self.s_init(annotations)), uh

    def sample_prev_y(self, k, ys_e, y_tm1_model=None, oracles=None):

        batch_size = ys_e.size(0)
        if wargs.ss_type is not None and ss_eps < 1. and (wargs.greed_sampling or wargs.bleu_sampling):

            if wargs.greed_sampling is True:
                if oracles is not None:     # joint word and sentence level
                    _seed = tc.zeros(batch_size, 1, requires_grad=False).bernoulli_()
                    if wargs.gpu_id is not None: _seed = _seed.cuda()
                    y_tm1_oracle = y_tm1_model * _seed + oracles[k] * (1. - _seed)
                    #y_tm1_oracle = y_tm1_model.data.mul_(_seed) + y_tm1_oracle.data.mul_(1. - _seed)
                    #wlog('joint word and sent ... ')
                else:
                    y_tm1_oracle = y_tm1_model  # word-level oracle (w/o w/ noise)
                    #wlog('word level oracle ... ')
            else:
                y_tm1_oracle = oracles[k]   # sentence-level oracle
                #wlog('sent level oracle ... ')

            #uval = tc.rand(batch_size, 1)    # different word and differet batch
            #if wargs.gpu_id: uval = uval.cuda()
            _g = tc.bernoulli( ss_eps * tc.ones(batch_size, 1) )   # pick gold with the probability of ss_eps
            _g = tc.tensor(_g, requires_grad=False)
            if wargs.gpu_id is not None: _g = _g.cuda()
            y_tm1 = ys_e[:, k, :] * _g + y_tm1_oracle * (1. - _g)
            #y_tm1 = schedule_sample_word(_h, _g, ss_eps, ys_e[k], y_tm1_oracle)
            #y_tm1 = ys_e[k].data.mul_(_g) + y_tm1_oracle.data.mul_(1. - _g)
        else:
            y_tm1 = ys_e[:, k, :]
            #g = self.sigmoid(self.w_gold(y_tm1) + self.w_hypo(y_tm1_oracle))
            #y_tm1 = g * y_tm1 + (1. - g) * y_tm1_oracle

        return y_tm1

    def step(self, s_tm1, xs_h, uh, y_tm1, xs_mask=None, y_mask=None):

        if not isinstance(y_tm1, tc.Tensor):
            if isinstance(y_tm1, int): y_tm1 = tc.tensor([y_tm1], dtype=tc.long, requires_grad=False)
            elif isinstance(y_tm1, list): y_tm1 = tc.tensor(y_tm1, dtype=tc.long, requires_grad=False)
            if wargs.gpu_id is not None: y_tm1 = y_tm1.cuda()
            _, y_tm1 = self.word_emb(y_tm1)

        #if xs_mask is not None:
        #    xs_mask = tc.tensor(xs_mask, requires_grad=False)
        #    if wargs.gpu_id is not None: xs_mask = xs_mask.cuda()

        state = self.lgru(y_tm1, s_tm1, y_mask)
        for layer_idx in range(self.n_layers - 1):
            state = self.tgrus[layer_idx](state, y_mask)
        # state: (batch_size, d_dec_hid)

        alpha, context = self.attention(state, xs_h, uh, xs_mask)
        # alpha:   [batch_size, n_head, key_len]
        # context: [batch_size, 2 * d_dec_hid]

        s_t = self.cond_lgru(context, state, y_mask)
        for layer_idx in range(self.n_layers - 1):
            s_t = self.cond_tgrus[layer_idx](s_t, y_mask)

        alpha = alpha[:, 0, :]  # get the attention of the first head, [batch_size, key_len]
        #alpha = alpha[:, 0, :].transpose(0, 1)  # get the attention of the first head, [key_len, batch_size]
        return context, s_t, y_tm1, alpha

    def forward(self, xs_h, ys, xs_mask, ys_mask, isAtt=False, ss_eps=1., oracles=None):

        s_tm1, uh = self.init_state(xs_h, xs_mask)
        tlen_batch_s, tlen_batch_y, tlen_batch_c = [], [], []
        batch_size, y_Lm1 = ys.size(0), ys.size(1)

        if isAtt is True: attends = []
        if ys.dim() == 3: ys_e = ys
        else: _, ys_e = self.word_emb(ys)   # (batch_size, max_tlen_batch - 1, d_trg_emb)

        sent_logit, y_tm1_model = [], ys_e[:, 0, :]
        for k in range(y_Lm1):

            y_tm1 = self.sample_prev_y(k, ys_e, y_tm1_model, oracles)
            context, s_tm1, _, alpha_ij = self.step(s_tm1, xs_h, uh, y_tm1, xs_mask, ys_mask[:, k])
            logit = self.step_out(s_tm1, y_tm1, context)
            sent_logit.append(logit)

            if wargs.ss_type is not None and ss_eps < 1. and wargs.greed_sampling is True:
                logit = self.classifier.get_a(logit, noise=wargs.greed_gumbel_noise)
                y_tm1_model = logit.max(-1)[1]
                _, y_tm1_model = self.word_emb(y_tm1_model)
                #wlog('word-level greedy sampling, noise {}'.format(wargs.greed_gumbel_noise))

            if isAtt is True: attends.append(alpha_ij)

        logit = tc.stack(sent_logit, dim=1)     # (batch_size, max_tlen_batch-1, d_dec_hid)
        logit = logit * ys_mask[:, :, None]  # !!!!

        results = (logit, tc.stack(attends, 0)) if isAtt is True else logit

        return results

    def step_out(self, s, y, c):

        # (max_tlen_batch - 1, batch_size, dec_hid_size)
        logit = self.s_transform(s) + self.y_transform(y) + self.c_transform(c)

        if self.max_out is True:
            if logit.dim() == 2:    # for decoding
                logit = logit.view(logit.size(0), logit.size(1)/2, 2)
            elif logit.dim() == 3:
                logit = logit.view(logit.size(0), logit.size(1), logit.size(2)/2, 2)
            logit = logit.max(-1)[0]

        logit = self.tanh(logit)

        if self.out_dropout_prob is not None and 0. < self.out_dropout_prob <= 1.0:
            logit = self.output_dropout(logit)

        return logit


