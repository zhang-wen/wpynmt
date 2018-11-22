from __future__ import division

import math
import torch as tc
import torch.nn as nn
import torch.nn.functional as F

import wargs
from tools.utils import *
from models.nn_utils import MaskSoftmax, MyLogSoftmax

class Classifier(nn.Module):

    def __init__(self, input_size, output_size, trg_word_emb=None, label_smoothing=0.,
                 emb_loss=False, bow_loss=False):

        super(Classifier, self).__init__()
        if emb_loss is True:
            assert trg_word_emb is not None, 'embedding loss needs target embedding'
            #self.trg_word_emb = trg_word_emb.we.weight
            self.trg_word_emb = trg_word_emb.we
            self.euclidean_dist = nn.PairwiseDistance(p=2, eps=1e-06, keepdim=True)
        self.emb_loss = emb_loss
        if bow_loss is True:
            wlog('using the bag of words loss')
            self.sigmoid = nn.Sigmoid()
            #self.softmax = MaskSoftmax()
        self.bow_loss = bow_loss

        self.map_vocab = nn.Linear(input_size, output_size, bias=True)
        if wargs.proj_share_weight is True:
            assert input_size == wargs.d_trg_emb
            wlog('copying weights of target word embedding into classifier')
            self.map_vocab.weight = trg_word_emb.we.weight
        self.log_prob = MyLogSoftmax(wargs.self_norm_alpha)

        assert 0. <= label_smoothing <= 1., 'label smoothing value should be in [0, 1]'
        wlog('NLL loss with label_smoothing: {}'.format(label_smoothing))
        if label_smoothing == 0.:
            weight = tc.ones(self.output_size)
            weight[PAD] = 0   # do not predict padding, same with ingore_index
            criterion = nn.NLLLoss(weight, ignore_index=PAD, reduction='sum')
            #self.criterion = nn.NLLLoss(weight, ignore_index=PAD, size_average=False)
        elif 0. < label_smoothing <= 1.:
            # all non-true labels are uniformly set to low-confidence
            self.smoothing_value = label_smoothing / (output_size - 2)
            one_hot = tc.full((output_size, ), self.smoothing_value)
            one_hot[PAD] = 0.
            self.register_buffer('one_hot', one_hot.unsqueeze(0))
            self.confidence = 1.0 - label_smoothing

        self.output_size = output_size
        self.softmax = MaskSoftmax()
        self.label_smoothing = label_smoothing

    def pred_map(self, logit, noise=None):

        logit = self.map_vocab(logit)

        if noise is not None:
            logit.data.add_( -tc.log(-tc.log(tc.Tensor(
                logit.size()).cuda().uniform_(0, 1) + epsilon) + epsilon) ) / noise

        return logit

    def logit_to_prob(self, logit, gumbel=None, tao=None):

        # (L, B)
        d1, d2, _ = logit.size()
        logit = self.pred_map(logit)
        if gumbel is None:
            p = self.softmax(logit)
        else:
            #print 'logit ..............'
            #print tc.max((logit < 1e+10) == False)
            #print 'gumbel ..............'
            #print tc.max((gumbel < 1e+10) == False)
            #print 'aaa ..............'
            #aaa = (gumbel.add(logit)) / tao
            #print tc.max((aaa < 1e+10) == False)
            p = self.softmax((gumbel.add(logit)) / tao)
        p = p.view(d1, d2, self.output_size)

        return p

    def smoothingXentLoss(self, pred_ll, target):

        # pred_ll (FloatTensor): batch_size*max_seq_len, n_classes
        # target  (LongTensor):  batch_size*max_seq_len
        if self.label_smoothing == 0.:
            # if label smoothing value is set to zero, the loss is equivalent to NLLLoss
            return self.criterion(ll, gold)
        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == PAD).unsqueeze(1), 0)
        #print pred_ll.size(), model_prob.size()
        xentropy = -(pred_ll * model_prob).sum()
        normalizing = -(self.confidence * math.log(self.confidence) + \
                        (self.output_size - 2) * self.smoothing_value * math.log(self.smoothing_value + 1e-20))

        return xentropy - normalizing

    def embeddingLoss(self, max_L, gold, gold_mask, bow=None, bow_mask=None):
        E, bow_L = self.trg_word_emb.size(1), bow.size(1)
        gold_emb = self.trg_word_emb(gold) * gold_mask[:, None]
        bow_emb = self.trg_word_emb(bow) * bow_mask[:, :, None]
        bow_emb = bow_emb[:,None,:,:].expand((-1, max_L, -1, -1)).contiguous().view(-1, E)
        gold_emb = gold_emb.reshape(batch_size, max_L, gold_emb.size(-1))[:,:,None,:].expand(
            (-1, -1, bow_L, -1)).contiguous().view(-1, E)
        dist = F.pairwise_distance(bow_emb, gold_emb, p=2, keepdim=True)
        dist = dist.reshape(batch_size, max_L, bow_L).sum(-1).view(-1)
        if gold_mask is not None: dist = dist.view(-1) * gold_mask
        pred_p_t = tc.gather(prob, dim=1, index=gold[:, None])
        if gold_mask is not None: pred_p_t = pred_p_t * gold_mask[:, None]
        return ( pred_p_t * dist[:, None] ).sum()

    def nll_loss(self, pred_2d, pred_3d, gold, gold_mask, bow=None, bow_mask=None, epo_idx=None):

        #print pred_2d.size(), pred_3d.size(), gold.size(), gold_mask.size(), bow.size(), bow_mask.size()
        batch_size, max_L, bow_L = pred_3d.size(0), pred_3d.size(1), bow.size(1)
        log_norm, prob, ll = self.log_prob(pred_2d)
        abs_logZ = (log_norm * gold_mask[:, None]).abs().sum()
        ll = ll * gold_mask[:, None]
        ce_loss = self.smoothingXentLoss(ll, gold)

        # embedding loss
        if self.emb_loss is True:
            emb_loss = self.embeddingLoss(max_L, gold, gold_mask, bow, bow_mask)
            loss = ce_loss + emb_loss
        elif self.bow_loss is True:
            gold_mask_3d = gold_mask.reshape(batch_size, max_L)[:,:,None]
            bow_prob = self.sigmoid((pred_3d * gold_mask_3d).sum(1))
            #bow_prob = self.softmax((pred_3d * gold_mask_3d).sum(1), gold_mask_3d)
            assert epo_idx is not None
            epo_idx = int(epo_idx[0, 0])
            bow_ll = tc.log(bow_prob + 1e-20)[:, None, :].expand(
                -1, bow_L, -1).contiguous().view(-1, bow_prob.size(-1))
            bow_ll = bow_ll * bow_mask.view(-1)[:, None]
            lambd = schedule_bow_lambda(epo_idx)
            loss = ( ce_loss / gold_mask.sum().item() ) + \
                    lambd * ( self.criterion(bow_ll, bow.view(-1)) / bow_mask.sum().item() )
        else:
            loss = ce_loss

        return loss, ce_loss, abs_logZ

    def forward(self, feed, gold=None, gold_mask=None, noise=None,
                bow=None, bow_mask=None, epo=None):

        # (batch_size, max_tlen_batch - 1, out_size)
        pred = self.pred_map(feed, noise)
        # decoding, if gold is None and gold_mask is None:
        if gold is None: return -self.log_prob(pred)[-1] if wargs.self_norm_alpha is None else -pred

        pred_vocab_3d = pred
        assert pred_vocab_3d.dim() == 3, 'error'
        if pred_vocab_3d.dim() == 3: pred = pred_vocab_3d.view(-1, pred_vocab_3d.size(-1))

        if gold.dim() == 2: gold, gold_mask = gold.view(-1), gold_mask.view(-1)
        # negative likelihood log
        loss, ce_loss, abs_logZ = self.nll_loss(pred, pred_vocab_3d, gold, gold_mask, bow, bow_mask, epo)

        # (max_tlen_batch - 1, batch_size, trg_vocab_size)
        ok_ytoks = (pred.max(dim=-1)[1]).eq(gold).masked_select(gold.ne(PAD)).sum()

        # total loss,  ok prediction count in one minibatch
        return loss, ce_loss, ok_ytoks, abs_logZ

    '''
    Compute the loss in shards for efficiency
        outputs: the predict outputs from the model
        gold: correct target sentences in current batch
    '''
    def snip_back_prop(self, outputs, gold, gold_mask, bow, bow_mask,
                       epo, shard_size=100, norm='sents'):

        # (batch_size, max_tlen_batch - 1, out_size)
        batch_nll, batch_ok_ytoks, batch_abs_logZ = 0, 0, 0
        epo = tc.ones_like(gold, requires_grad=False) * epo
        normalization = gold_mask.sum().item() if norm == 'tokens' else outputs.size(1)
        shard_state = { 'feed': outputs, 'gold': gold, 'gold_mask': gold_mask, 'bow': bow,
                       'bow_mask':bow_mask, 'epo': epo }

        for shard in shards(shard_state, shard_size):
            loss, nll, ok_ytoks, abs_logZ = self(**shard)
            batch_nll += nll.item()
            batch_ok_ytoks += ok_ytoks.item()
            batch_abs_logZ += abs_logZ.item()
            loss.div(float(normalization)).backward(retain_graph=True)

        return batch_nll, batch_ok_ytoks, batch_abs_logZ

def filter_shard_state(state, shard_size=None):
    for k, v in state.items():
        if shard_size is None:
            yield k, v

        if v is not None:
            v_split = []
            if isinstance(v, tc.Tensor):
                for v_chunk in tc.split(v, shard_size):
                    v_chunk = v_chunk.data.clone()
                    v_chunk.requires_grad = v.requires_grad
                    v_split.append(v_chunk)
            yield k, (v, v_split)


def shards(state, shard_size, eval_only=False):
    '''
    Args:
        state: A dictionary which corresponds to the output
               values for those keys are Tensor-like or None.
        shard_size: The maximum size of the shards yielded by the model.
        eval_only: if True, only yield the state, nothing else.
              otherwise, yield shards.
    Yields:
        each yielded shard is a dict.
    side effect:
        after the last shard, this function does back-propagation.
    '''
    if eval_only:
        yield filter_shard_state(state)
    else:
        # non_none: the subdict of the state dictionary where the values are not None
        non_none = dict(filter_shard_state(state, shard_size))

        # Now, the iteration:
        # state is a dictionary of sequences of tensor-like but we
        # want a sequence of dictionaries of tensors.
        # First, unzip the dictionary into a sequence of keys and a
        # sequence of tensor-like sequences.
        keys, values = zip(*((k, [v_chunk for v_chunk in v_split])
                             for k, (_, v_split) in non_none.items()))

        # Now, yield a dictionary for each shard. The keys are always
        # the same. values is a sequence of length #keys where each
        # element is a sequence of length #shards. We want to iterate
        # over the shards, not over the keys: therefore, the values need
        # to be re-zipped by shard and then each shard can be paired
        # with the keys.
        for shard_tensors in zip(*values):
            yield dict(zip(keys, shard_tensors))

        # Assumed backprop'd
        variables = []
        for k, (v, v_split) in non_none.items():
            if isinstance(v, tc.Tensor) and state[k].requires_grad:
                variables.extend(zip(tc.split(state[k], shard_size),
                                     [v_chunk.grad for v_chunk in v_split]))
        inputs, grads = zip(*variables)
        tc.autograd.backward(inputs, grads)

