from __future__ import division

import sys
import copy
import numpy as np
import torch as tc
from torch.autograd import Variable

import wargs
from tools.utils import *

import time

class Nbs(object):

    def __init__(self, model, tvcb_i2w, k=10, ptv=None, noise=False, print_att=False):

        self.model = model
        self.decoder = model.decoder

        self.tvcb_i2w = tvcb_i2w
        self.k = k
        self.ptv = ptv
        self.noise = noise
        self.xs_mask = None
        self.print_att = print_att

        self.C = [0] * 4
        '''
        prob_projection = nn.LogSoftmax()
        self.model.prob_projection = prob_projection.cuda()
        #self.print_att = False
        '''

    def beam_search_trans(self, input_data, x_mask=None):

        self.beam, self.hyps, self.B = [], [], 1
        #self.attent_probs = [] if self.print_att is True else None
        self.attent_probs = [[] for _ in range(self.B)] if self.print_att is True else None
        self.batch_tran_cands = [[] for _ in range(self.B)]
        #print '-------------------- one sentence ............'
        if isinstance(input_data, list):
            self.srcL = len(s_list)
            s_tensor = tc.Tensor(s_list).long().unsqueeze(-1)
        elif isinstance(input_data, tuple):
            # input_data: (idxs, tsrcs, tspos, lengths, src_mask)
            if len(input_data) == 5:
                _, s_tensor, src_pos, lens, src_mask = input_data
                self.src_seq_BL, src_pos_BL = s_tensor.t(), src_pos.t()
            elif len(input_data) == 2:
                self.src_seq_BL, src_pos_BL = input_data
            elif len(input_data) == 8:
                _, s_tensor, src_pos, _, _, _, _, _ = input_data
                self.src_seq_BL, src_pos_BL = s_tensor.t(), src_pos.t()
            assert self.src_seq_BL.size(0) == 1, 'Unsupported for batch decoding ... '
            self.srcL = self.src_seq_BL.size(1)
        self.maxL = wargs.max_seq_len
        if x_mask is None:
            x_mask = tc.ones((self.srcL, 1))
            x_mask = Variable(x_mask, requires_grad=False, volatile=True).cuda()

        self.enc_src0, _, _ = self.model.encoder(self.src_seq_BL, src_pos_BL)
        #self.enc_src0 = self.model.encoder(self.src_seq_BL, src_pos_BL)[0]
        init_beam(self.beam, cnt=self.maxL, cp=True)

        if not wargs.with_batch: best_trans, best_loss = self.search()
        elif wargs.ori_search:   best_trans, best_loss = self.ori_batch_search()
        else:                    self.batch_search()
        # best_trans w/o <bos> and <eos> !!!

        #batch_tran_cands: [(trans, loss, attend)]
        for bidx in range(self.B):
            debug([ (a[0], a[1]) for a in self.batch_tran_cands[bidx] ])
            best_trans, best_loss = self.batch_tran_cands[bidx][0][0], self.batch_tran_cands[bidx][0][1]
            debug('Src[{}], maskL[{}], hyp (w/o EOS)[{}], maxL[{}], loss[{}]'.format(
                x_mask[:, bidx].sum().data[0], self.srcL, len(best_trans), self.maxL, best_loss))
        debug('Average Merging Rate [{}/{}={:6.4f}]'.format(self.C[1], self.C[0], self.C[1] / self.C[0]))
        debug('Average location of bp [{}/{}={:6.4f}]'.format(self.C[3], self.C[2], self.C[3] / self.C[2]))
        #debug('Step[{}] stepout[{}]'.format(*self.C[4:]))

        return self.batch_tran_cands

    ##################################################################

    # Wen Zhang: beam search, no batch

    ##################################################################
    def search(self):

        for i in range(1, self.maxL + 1):

            prevb = self.beam[i - 1]
            preb_sz = len(prevb)
            cnt_bp = (i >= 2)
            if cnt_bp: self.C[0] += preb_sz
            cands = []
            for j in xrange(preb_sz):  # size of last beam
                # (45.32, (beam, trg_nhids), -1, 0)
                accum_im1, s_im1, y_im1, bp_im1 = self.beam[i - 1][j]

                a_i, s_i, y_im1, _, _, _ = self.decoder.step(s_im1, self.enc_src0, self.uh0, y_im1)
                self.C[2] += 1
                logit = self.decoder.step_out(s_i, y_im1, a_i)
                self.C[3] += 1

                next_ces = self.model.decoder.classifier(logit)
                next_ces = next_ces.cpu().data.numpy()
                next_ces_flat = next_ces.flatten()    # (1,vocsize) -> (vocsize,)
                ranks_idx_flat = part_sort(next_ces_flat, self.k - len(self.hyps))
                k_avg_loss_flat = next_ces_flat[ranks_idx_flat]  # -log_p_y_given_x

                accum_i = accum_im1 + k_avg_loss_flat
                cands += [(accum_i[idx], s_i, wid, j) for idx, wid in enumerate(ranks_idx_flat)]

            k_ranks_flat = part_sort(np.asarray(
                [cand[0] for cand in cands] + [np.inf]), self.k - len(self.hyps))
            k_sorted_cands = [cands[r] for r in k_ranks_flat]

            for b in k_sorted_cands:
                if cnt_bp: self.C[1] += (b[-1] + 1)
                if b[-2] == EOS:
                    if wargs.len_norm: self.hyps.append(((b[0] / i), b[0]) + b[-2:] + (i,))
                    else: self.hyps.append((b[0], ) + b[-2:] + (i, ))
                    debug('Gen hypo {}'.format(self.hyps[-1]))
                    if len(self.hyps) == self.k:
                        # output sentence, early stop, best one in k
                        debug('Early stop! see {} hyps ending with EOS.'.format(self.k))
                        sorted_hyps = sorted(self.hyps, key=lambda tup: tup[0])
                        for hyp in sorted_hyps: debug('{}'.format(hyp))
                        best_hyp = sorted_hyps[0]
                        debug('Best hyp length (w/ EOS)[{}]'.format(best_hyp[-1]))
                        return back_tracking(self.beam, best_hyp)
                # should calculate when generate item in current beam
                else: self.beam[i].append(b)

            debug('beam {} ----------------------------'.format(i))
            for b in self.beam[i]: debug(b[0:1] + b[2:])    # do not output state

        return back_tracking(self.beam, self.no_early_best())

    #@exeTime
    def batch_search(self):

        #encoded_src0 = self.enc_src0
        encoded_src0 = self.enc_src0[-1]
        # s0: (1, trg_nhids), encoded_src0: (srcL, 1, src_nhids*2), uh0: (srcL, 1, align_size)
        enc_size = encoded_src0.size(-1)
        L, align_size = self.srcL, wargs.align_size
        hyp_scores = np.zeros(1).astype('float32')
        delete_idx, prevb_id = None, None
        #batch_adj_list = [range(self.srcL) for _ in range(self.k)]

        debug('\nBeam-{} {}'.format(0, '-'*20))
        for b in self.beam[0][0]:    # do not output state
            if wargs.dynamic_cyk_decoding is True:
                debug(b[0:1] + (b[1][0], b[1][1], b[1][2].data.int().tolist()) + b[-2:])
            else: debug(b[0:1] + b[-2:])

        btg_xs_h, btg_uh, btg_xs_mask = None, None, None
        if wargs.dynamic_cyk_decoding is True: btg_xs_h = encoded_src0

        def track_ys(cur_bidx):
            y_part_seqs = []
            for b in self.beam[cur_bidx - 1][0]:
                seq, bp = [b[-2]], b[-1]
                for i in reversed(xrange(0, cur_bidx - 1)):
                    _, _, _, _, w, backptr = self.beam[i][0][bp]
                    seq.append(w)
                    bp = backptr
                y_part_seqs.append(seq[::-1])
            return y_part_seqs

        debug('Last layer output of encoder: {}'.format(encoded_src0.size()))
        for i in range(1, self.maxL + 1):

            debug('\n{} Step-{} {}'.format('#'*20, i, '#'*20))
            prevb = self.beam[i - 1][0]
            preb_sz = len(prevb)
            cnt_bp = (i >= 2)
            if cnt_bp: self.C[0] += preb_sz

            len_dec_seq = i
            # -- Preparing decoded pos seq -- #
            y_part_pos = tc.arange(1, len_dec_seq + 1).unsqueeze(0) # (1, seq)
            y_part_pos = y_part_pos.repeat(preb_sz, 1)    # (beam, src_L)
            y_part_pos = Variable(y_part_pos.type(tc.LongTensor), volatile=True)
            # -- Preparing decoded data seq -- #
            y_part_seqs = track_ys(i) # (preb_sz, trg_part_L)
            y_part_seqs = tc.Tensor(y_part_seqs).view(-1, len_dec_seq)
            y_part_seqs = Variable(y_part_seqs.type(tc.LongTensor), volatile=True)
            if wargs.gpu_id is not None:
                y_part_pos = y_part_pos.cuda()
                y_part_seqs = y_part_seqs.cuda()
            # encoded_src0: (mb_size, len_q, d_model)
            #print self.src_seq_BL.size(), encoded_src0.size()
            src_seq_BL = self.src_seq_BL.contiguous().view(-1, L).expand(preb_sz, L)
            # (1, srcL, src_nhids) -> (preb_sz, srcL, src_nhids)
            #enc_srcs = [self.enc_src0[l_idx].view(-1, L, enc_size).expand(preb_sz, L, enc_size) \
            #            for l_idx in range(len(self.enc_src0))]
            #print src_seq_BL.size(), enc_src.size()
            enc_srcs = encoded_src0.expand(preb_sz, L, enc_size)

            #print y_part_seqs.size(), y_part_pos.size(), self.src_seq_BL.size(), enc_src.size()
            # -- Decoding -- #
            debug('Whole x seq: {}'.format(src_seq_BL.size()))
            debug('Part y seq: {}'.format(y_part_seqs.size()))
            debug('Part y pos: {}'.format(y_part_pos.size()))
            #print enc_srcs
            dec_output, dec_slf_attn, dec_enc_attn, alpha_ij = self.model.decoder(
                y_part_seqs, y_part_pos, src_seq_BL, enc_srcs)
            #dec_output = self.model.decoder(
            #    y_part_seqs, y_part_pos, src_seq_BL, enc_srcs)[0]
            debug('History decoder output: {}'.format(dec_output.size()))
            # (preb_sz, part_Len, d_model) -> (preb_sz, d_model)
            dec_output = dec_output[:, -1, :] # (preb_sz, d_model) previous decoder hidden state
            debug('Previous decoder output: {}'.format(dec_output.size()))
            alpha_ij = alpha_ij[:, -1, :].permute(1, 0)    # (B, trgL, srcL) -> (srcL, B)
            if self.attent_probs is not None:
                #print 'alpha_ij: ', alpha_ij.size()
                self.attent_probs[0].append(alpha_ij)
            self.C[2] += 1
            self.C[3] += 1
            debug('For beam[{}], pre-beam ids: {}'.format(i - 1, prevb_id))
            next_ces = self.model.decoder.classifier(dec_output)
            '''
            dec_output = self.model.tgt_word_proj(dec_output)
            next_ces = -self.model.prob_projection(dec_output)
            '''
            next_ces = next_ces.cpu().data.numpy()
            #next_ces = -next_scores if self.ifscore else self.fn_ce(next_scores)
            cand_scores = hyp_scores[:, None] + next_ces
            cand_scores_flat = cand_scores.flatten()
            ranks_flat = part_sort(cand_scores_flat, self.k - len(self.hyps))
            voc_size = next_ces.shape[1]
            prevb_id = ranks_flat // voc_size
            word_indices = ranks_flat % voc_size
            costs = cand_scores_flat[ranks_flat]

            debug('For beam[{}], pre-beam ids: {}'.format(i, prevb_id))

            tp_bid = tc.from_numpy(prevb_id).cuda() if wargs.gpu_id else tc.from_numpy(prevb_id)
            delete_idx, next_beam_cur_sent = [], []
            for _j, b in enumerate(zip(costs, word_indices, prevb_id)):
                bp = b[-1]
                if wargs.len_norm == 0: score = (b[0], None)
                elif wargs.len_norm == 1: score = (b[0] / i, b[0])
                elif wargs.len_norm == 2:   # alpha length normal
                    lp, cp = lp_cp(bp, i, 0, self.beam)
                    score = (b[0] / lp + cp, b[0])
                if cnt_bp: self.C[1] += (bp + 1)
                if b[-2] == EOS:
                    delete_idx.append(b[-1])
                    debug(score)
                    self.hyps.append(score + (None, ) + b[-2:] + (i, ))
                    debug('Gen hypo {}'.format(self.hyps[-1]))
                    # because i starts from 1, so the length of the first beam is 1, no <bos>
                    if len(self.hyps) == self.k:
                        # output sentence, early stop, best one in k
                        debug('Early stop! see {} hyps ending with EOS.'.format(self.k))
                        sorted_hyps = sorted(self.hyps, key=lambda tup: tup[0])
                        for hyp in sorted_hyps: debug('{}'.format(hyp))
                        best_hyp = sorted_hyps[0]
                        debug('Best hyp length (w/ EOS)[{}]'.format(best_hyp[-1])) # w/ eos w/o bos
                        self.batch_tran_cands[0] = [back_tracking(self.beam, 0, hyp, \
                                                self.attent_probs[0] if self.attent_probs is not None \
                                                                  else None) for hyp in sorted_hyps]
                        return
                # should calculate when generate item in current beam
                else:
                    if wargs.len_norm == 2: next_beam_cur_sent.append((b[0], alpha_ij[:, bp], None, 0) + b[1:])
                    else: next_beam_cur_sent.append((b[0], None, None, 0) + b[1:])

            self.beam[i] = [ next_beam_cur_sent ]

            debug('\n{} Beam-{} {}'.format('-'*20, i, '-'*20))
            for b in self.beam[i][0]:    # do not output state
                if wargs.dynamic_cyk_decoding is True:
                    debug(b[0:1] + (b[1][0], b[1][1], b[1][2].data.int().tolist()) + b[-2:])
                else: debug(b[0:1] + b[-2:])
            hyp_scores = np.array([b[0] for b in self.beam[i][0]])

        # no early stop, back tracking
        #return back_tracking(self.beam, 0, self.no_early_best(), self.attent_probs)
        self.no_early_best()

    def no_early_best(self):

        # no early stop, back tracking
        debug('==Start== No early stop ...')
        if len(self.hyps) == 0:
            debug('No early stop, no hyp with EOS, select k hyps length {} '.format(self.maxL))
            best_hyp = self.beam[self.maxL][0][0]
            if wargs.len_norm == 0: score = (best_hyp[0], None)
            elif wargs.len_norm == 1: score = (best_hyp[0] / self.maxL, best_hyp[0])
            elif wargs.len_norm == 2:   # alpha length normal
                lp, cp = lp_cp(best_hyp[-1], self.maxL, 0, self.beam)
                score = (best_hyp[0] / lp + cp, best_hyp[0])
            self.hyps.append(score + best_hyp[-3:] + (self.maxL, ))
        else:
            debug('No early stop, no enough {} hyps with EOS, select the best '
                      'one from {} hyps.'.format(self.k, len(self.hyps)))
        sorted_hyps = sorted(self.hyps, key=lambda tup: tup[0])
        for hyp in sorted_hyps: debug('{}'.format(hyp))
        debug('Best hyp length (w/ EOS)[{}]'.format(sorted_hyps[0][-1]))
        self.batch_tran_cands[0] = [back_tracking(self.beam, 0, hyp, \
                    self.attent_probs[0] if self.attent_probs is not None \
                                            else None) for hyp in sorted_hyps]

    def ori_batch_search(self):

        sample = []
        sample_score = []

        live_k = 1
        dead_k = 0

        hyp_samples = [[]] * live_k
        hyp_scores = numpy.zeros(live_k).astype('float32')
        hyp_states = []

        # s0: (1, trg_nhids), enc_src0: (srcL, 1, src_nhids*2), uh0: (srcL, 1, align_size)
        L, enc_size, align_size = self.srcL, self.enc_src0.size(2), self.uh0.size(2)

        s_im1, y_im1 = self.s0, [BOS]  # indicator for the first target word (bos target)
        preb_sz = 1

        for ii in xrange(self.maxL):

            cnt_bp = (ii >= 1)
            if cnt_bp: self.C[0] += preb_sz
            # (src_sent_len, 1, 2*src_nhids) -> (src_sent_len, live_k, 2*src_nhids)
            enc_src = self.enc_src0.view(L, -1, enc_size).expand(L, live_k, enc_size)
            uh = self.uh0.view(L, -1, align_size).expand(L, live_k, align_size)

            #c_i, s_i = self.decoder.step(c_im1, enc_src, uh, y_im1)
            a_i, s_im1, y_im1, _ = self.decoder.step(s_im1, enc_src, uh, y_im1)
            self.C[2] += 1
            # (preb_sz, out_size)
            # logit = self.decoder.logit(s_i)
            logit = self.decoder.step_out(s_im1, y_im1, a_i)
            self.C[3] += 1
            next_ces = self.model.decoder.classifier(logit)
            next_ces = next_ces.cpu().data.numpy()
            #cand_scores = hyp_scores[:, None] - numpy.log(next_scores)
            cand_scores = hyp_scores[:, None] + next_ces
            cand_flat = cand_scores.flatten()
            # ranks_flat = cand_flat.argsort()[:(k-dead_k)]
            # we do not need to generate k candidate here, because we just need to generate k-dead_k
            # more candidates ending with eos, so for each previous candidate we just need to expand
            # k-dead_k candidates
            ranks_flat = part_sort(cand_flat, self.k - dead_k)
            # print ranks_flat, cand_flat[ranks_flat[1]], cand_flat[ranks_flat[8]]

            voc_size = next_ces.shape[1]
            trans_indices = ranks_flat // voc_size
            word_indices = ranks_flat % voc_size
            costs = cand_flat[ranks_flat]

            new_hyp_samples = []
            new_hyp_scores = numpy.zeros(self.k - dead_k).astype('float32')
            new_hyp_states = []

            for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
                if cnt_bp: self.C[1] += (ti + 1)
                new_hyp_samples.append(hyp_samples[ti] + [wi])
                new_hyp_scores[idx] = costs[idx]
                new_hyp_states.append(s_im1[ti])

            # check the finished samples
            new_live_k = 0
            hyp_samples = []
            hyp_scores = []
            hyp_states = []
            # current beam, if the hyposise ends with eos, we do not
            for idx in xrange(len(new_hyp_samples)):
                if new_hyp_samples[idx][-1] == EOS:
                    sample.append(new_hyp_samples[idx])
                    sample_score.append(new_hyp_scores[idx])
                    # print new_hyp_scores[idx], new_hyp_samples[idx]
                    dead_k += 1
                else:
                    new_live_k += 1
                    hyp_samples.append(new_hyp_samples[idx])
                    hyp_scores.append(new_hyp_scores[idx])
                    hyp_states.append(new_hyp_states[idx])
            hyp_scores = numpy.array(hyp_scores)
            live_k = new_live_k

            if new_live_k < 1: break
            if dead_k >= self.k: break

            preb_sz = len(hyp_states)
            y_im1 = [w[-1] for w in hyp_samples]
            s_im1 = tc.stack(hyp_states, dim=0)

        if live_k > 0:
            for idx in xrange(live_k):
                sample.append(hyp_samples[idx])
                sample_score.append(hyp_scores[idx])

        if wargs.len_norm:
            lengths = numpy.array([len(s) for s in sample])
            sample_score = sample_score / lengths
        sidx = numpy.argmin(sample_score)

        return sample[sidx], sample_score[sidx]

