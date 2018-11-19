from __future__ import division

import os
import sys
import math
import time
import subprocess

import numpy as np
import torch as tc

import wargs
from tools.utils import *
from searchs.nbs import Nbs
from translate import Translator

class Trainer(object):

    def __init__(self, model, train_data, vocab_data, optim, valid_data=None, tests_data=None):

        self.model, self.optim = model, optim
        if isinstance(model, tc.nn.DataParallel): self.classifier = model.module.classifier
        else: self.classifier = model.classifier

        self.sv, self.tv = vocab_data['src'].idx2key, vocab_data['trg'].idx2key
        self.train_data, self.valid_data, self.tests_data = train_data, valid_data, tests_data
        self.max_epochs, self.start_epoch = wargs.max_epochs, wargs.start_epoch

        wlog('self-normalization alpha -> {}'.format(wargs.self_norm_alpha))

        self.n_look = wargs.n_look
        assert self.n_look <= wargs.batch_size, 'eyeball count > batch size'
        self.n_batches = len(train_data)    # [low, high)
        self.look_xs, self.look_ys = None, None
        if wargs.fix_looking:
            look_bidx = tc.randperm(self.n_batches)[-1]
            wlog('randomly look {} samples in the {}th/{} batch'.format(
                self.n_look, look_bidx, self.n_batches))
            _, xs, y_for_files, _, _, _, _, _ = train_data[look_bidx]
            ys, _batch_size = y_for_files[0], xs.size(0)
            rand_rows = np.random.choice(_batch_size, self.n_look, replace=False)
            self.look_xs = tc.LongTensor(self.n_look, xs.size(1))
            self.look_xs.fill_(PAD)
            self.look_ys = tc.LongTensor(self.n_look, ys.size(1))
            self.look_ys.fill_(PAD)
            for _idx in xrange(self.n_look):
                self.look_xs[_idx, :] = xs[rand_rows[_idx], :]
                self.look_ys[_idx, :] = ys[rand_rows[_idx], :]
        self.look_tor = Translator(self.model, self.sv, self.tv)
        self.n_eval = 0

        self.snip_size, self.trunc_size = wargs.snip_size, wargs.trunc_size
        self.grad_accum_count = wargs.grad_accum_count
        self.norm_type = wargs.normalization

        self.epoch_shuffle_train = wargs.epoch_shuffle_train
        self.epoch_shuffle_batch = wargs.epoch_shuffle_batch
        self.ss_eps_cur = wargs.ss_eps_begin
        if wargs.ss_type is not None:
            wlog('word-level optimizing bias between training and decoding ...')
            if wargs.bleu_sampling is True: wlog('sentence-level optimizing ...')
            wlog('schedule sampling value {}'.format(self.ss_eps_cur))
            if self.ss_eps_cur < 1. and wargs.bleu_sampling:
                self.sampler = Nbs(self.model, self.tv, k=3, noise=wargs.bleu_gumbel_noise,
                                   batch_sample=True)
        if self.grad_accum_count > 1:
            assert(self.trunc_size == 0), 'to accumulate grads, disable target sequence truncating'

    def accum_matrics(self, batch_size, xtoks, ytoks, nll, ok_ytoks, logZ):

        self.look_sents += batch_size
        self.e_sents += batch_size
        self.look_nll += nll
        self.look_ok_ytoks += ok_ytoks
        self.e_nll += nll
        self.e_ok_ytoks += ok_ytoks
        self.look_xtoks += xtoks
        self.look_ytoks += ytoks
        self.e_ytoks += ytoks
        self.look_batch_logZ += logZ
        self.e_batch_logZ += logZ

    def grad_accumulate(self, real_batches, epo, norm_type='sents'):

        if self.grad_accum_count > 1:
            self.model.zero_grad()

        for batch in real_batches:

            # (batch_size, max_slen_batch)
            _, xs, y_for_files, bows, x_lens, xs_mask, y_mask_for_files, bows_mask = batch
            _batch_size = xs.size(0)
            ys, ys_mask = y_for_files[0], y_mask_for_files[0]
            #wlog('x: {}, x_mask: {}, y: {}, y_mask: {}'.format(
            #    xs.size(), xs_mask.size(), ys.size(), ys_mask.size()))
            if bows is not None:
                bows, bows_mask = bows[0], bows_mask[0]
                #wlog('bows: {}, bows_mask: {})'.format(bows.size(), bows_mask.size()))
            _xtoks = xs.data.ne(PAD).sum().item()
            assert _xtoks == x_lens.data.sum().item()
            _ytoks = ys[1:].data.ne(PAD).sum().item()

            ys_len = ys.size(1)
            # Truncated BPTT
            trunc_size = self.trunc_size if self.trunc_size else ys_len

            for j in range(0, ys_len - 1, trunc_size):
                # 1. Create truncated target.
                part_ys = ys[:, j : j + trunc_size]
                part_ys_mask = ys_mask[:, j : j + trunc_size]
                # 2. F-prop all but generator.
                if self.grad_accum_count == 1: self.model.zero_grad()
                # exclude last target word from inputs
                logits, alphas = self.model(xs, part_ys[:, :-1], xs_mask, part_ys_mask[:, :-1])
                # (batch_size, max_tlen_batch - 1, out_size)

                gold, gold_mask = part_ys[:, 1:].contiguous(), part_ys_mask[:, 1:].contiguous()
                # 3. Compute loss in shards for memory efficiency.
                _nll, _ok_ytoks, _logZ = self.classifier.snip_back_prop(
                    logits, gold, gold_mask, bows, bows_mask, epo, self.snip_size, self.norm_type)

            self.accum_matrics(_batch_size, _xtoks, _ytoks, _nll, _ok_ytoks, _logZ)
        # 3. Update the parameters and statistics.
        self.optim.step()
        tc.cuda.empty_cache()

    def look_samples(self, bidx, batch):

        if bidx % wargs.look_freq == 0:

            look_start = time.time()
            self.model.eval()   # affect the dropout !!!
            if self.look_xs and self.look_ys:
                _xs, _ys = self.look_xs, self.look_ys
            else:
                _, xs, y_for_files, _, _, _, _, _ = batch
                ys = y_for_files[0]
                # (batch_size, max_len_batch)
                rand_bids = np.random.choice(xs.size(0), self.n_look, replace=False)
                _xs, _ys = xs[rand_bids], ys[rand_bids]
            self.look_tor.trans_samples(_xs, _ys)
            wlog('')
            self.look_spend = time.time() - look_start
            self.model.train()

    def try_valid(self, epo, e_bidx, n_steps):

        if wargs.epoch_eval is not True and n_steps > wargs.eval_valid_from and \
           n_steps % wargs.eval_valid_freq == 0:
            eval_start = time.time()
            self.n_eval += 1
            wlog('\nAmong epo, batch [{}], [{}] eval save model ...'.format(e_bidx, self.n_eval))
            bleu = self.mt_eval(epo, e_bidx)
            self.optim.update_learning_rate(bleu, epo)
            self.eval_spend = time.time() - eval_start

    def mt_eval(self, eid, bid):

        state_dict = { 'model': self.model.state_dict(), 'epoch': eid, 'batch': bid, 'optim': self.optim }

        if wargs.save_one_model: model_file = '{}.pt'.format(wargs.model_prefix)
        else: model_file = '{}_e{}_upd{}.pt'.format(wargs.model_prefix, eid, bid)
        tc.save(state_dict, model_file)
        wlog('Saving temporary model in {}'.format(model_file))

        self.model.eval()

        tor0 = Translator(self.model, self.sv, self.tv, print_att=wargs.print_att)
        bleu = tor0.trans_eval(self.valid_data, eid, bid, model_file, self.tests_data)

        self.model.train()

        return bleu

    def train(self):

        wlog('start training ... ')
        train_start = time.time()
        wlog('\n' + '#' * 120 + '\n' + '#' * 30 + ' Start Training ' + '#' * 30 + '\n' + '#' * 120)
        batch_oracles, _checks, accum_batches, real_batches = None, None, 0, []
        self.model.train()

        for epo in range(self.start_epoch, self.max_epochs + 1):

            wlog('\n{} Epoch [{}/{}] {}'.format('$'*30, epo, self.max_epochs, '$'*30))
            # shuffle the training data for each epoch
            if self.epoch_shuffle_train: self.train_data.shuffle()

            self.e_nll, self.e_ytoks, self.e_ok_ytoks, self.e_batch_logZ, self.e_sents \
                    = 0, 0, 0, 0, 0
            self.look_nll, self.look_ytoks, self.look_ok_ytoks, self.look_batch_logZ, \
                    self.look_sents = 0, 0, 0, 0, 0
            self.look_xtoks, self.look_spend, b_counter, eval_spend = 0, 0, 0, 0
            epo_start = show_start = time.time()
            if self.epoch_shuffle_batch: shuffled_bidx = tc.randperm(self.n_batches)

            for bidx in range(self.n_batches):

                b_counter += 1
                e_bidx = shuffled_bidx[bidx] if self.epoch_shuffle_batch else bidx
                if wargs.ss_type is not None and self.ss_eps_cur < 1. and wargs.bleu_sampling:
                    batch_beam_trgs = self.sampler.beam_search_trans(xs, xs_mask, ys_mask)
                    batch_beam_trgs = [list(zip(*b)[0]) for b in batch_beam_trgs]
                    #wlog(batch_beam_trgs)
                    batch_oracles = batch_search_oracle(batch_beam_trgs, ys[1:], ys_mask[1:])
                    #wlog(batch_oracles)
                    batch_oracles = batch_oracles[:-1].cuda()
                    batch_oracles = self.model.decoder.trg_lookup_table(batch_oracles)

                batch = self.train_data[e_bidx]
                real_batches.append(batch)
                accum_batches += 1
                if accum_batches == self.grad_accum_count:

                    self.grad_accumulate(real_batches, epo)
                    current_steps = self.optim.n_current_steps
                    accum_batches, real_batches = 0, []
                    grad_checker(self.model, _checks)
                    if current_steps % wargs.display_freq == 0:
                        #print self.look_ok_ytoks, self.look_nll, self.look_ytoks, self.look_nll/self.look_ytoks
                        ud = time.time() - show_start - self.look_spend - eval_spend
                        wlog(
                            'Epo:{:>2}/{:>2} |[{:^5}/{} {:^5}] |acc:{:5.2f}% |nll:{:4.2f}'
                            ' |w-ppl:{:4.2f} |w(s)-logZ|:{:.2f}({:.2f}) '
                            ' |x(y)/s:{:>4}({:>4})/{}={}({}) |x(y)/sec:{}({}) |lr:{:7.6f}'
                            ' |{:4.2f}s/{:4.2f}m'.format(
                                epo, self.max_epochs, b_counter, self.n_batches, current_steps,
                                (self.look_ok_ytoks / self.look_ytoks) * 100,
                                self.look_nll / self.look_ytoks,
                                math.exp(self.look_nll / self.look_ytoks),
                                #math.exp(self.look_nll),
                                self.look_batch_logZ / self.look_ytoks,
                                self.look_batch_logZ / self.look_sents, self.look_xtoks,
                                self.look_ytoks, self.look_sents,
                                int(round(self.look_xtoks / self.look_sents)),
                                int(round(self.look_ytoks / self.look_sents)),
                                int(round(self.look_xtoks / ud)), int(round(self.look_ytoks / ud)),
                                self.optim.learning_rate, ud, (time.time() - train_start) / 60.)
                        )
                        self.look_nll, self.look_xtoks, self.look_ytoks, self.look_ok_ytoks, \
                                self.look_batch_logZ, self.look_sents = 0, 0, 0, 0, 0, 0
                        self.look_spend, eval_spend = 0, 0
                        show_start = time.time()

                    self.look_samples(current_steps, batch)
                    self.try_valid(epo, e_bidx, current_steps)

            avg_epo_acc, avg_epo_nll = self.e_ok_ytoks/self.e_ytoks, self.e_nll/self.e_ytoks
            wlog('\nEnd epoch [{}]'.format(epo))
            wlog('avg. w-acc: {:4.2f}%, w-nll: {:4.2f}, w-ppl: {:4.2f}'.format(
                avg_epo_acc * 100, avg_epo_nll, math.exp(avg_epo_nll)))
            wlog('avg. |w-logZ|: {:.2f}/{}={:.2f} |s-logZ|: {:.2f}/{}={:.2f}'.format(
                self.e_batch_logZ, self.e_ytoks, self.e_batch_logZ / self.e_ytoks,
                self.e_batch_logZ, self.e_sents, self.e_batch_logZ / self.e_sents))
            wlog('batch [{}], [{}] eval save model ...'.format(e_bidx, self.n_eval))
            if wargs.epoch_eval is True:
                bleu = self.mt_eval(epo, e_bidx)
                self.optim.update_learning_rate(bleu, epo)
            # decay the probability value epslion of scheduled sampling per batch
            if wargs.ss_type is not None: ss_eps_cur = schedule_sample_eps_decay(epo, ss_eps_cur)   # start from 1
            epo_time_consume = time.time() - epo_start
            wlog('Consuming: {:4.2f}s'.format(epo_time_consume))

        wlog('Finish training, comsuming {:6.2f} hours'.format((time.time() - train_start) / 3600))
        wlog('Congratulations!')

