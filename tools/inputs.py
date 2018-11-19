from __future__ import division

import math
import wargs
import torch as tc
from utils import *

class Input(object):

    def __init__(self, x_list, y_list, batch_size, bow=False, batch_sort=False, prefix=None, printlog=True):

        self.x_list = x_list
        self.n_sent = len(x_list)
        self.B = batch_size
        self.gpu_id = wargs.gpu_id
        self.batch_sort = batch_sort
        self.bow = bow

        if y_list is not None:
            self.y_list_files = y_list
            # [sent0:[ref0, ref1, ...], sent1:[ref0, ref1, ... ], ...]
            assert self.n_sent == len(y_list)
            if printlog is True:
                wlog('Bilingual: batch size {}, Sort in batch? {}'.format(self.B, batch_sort))
            else:
                debug('Bilingual: batch size {}, Sort in batch? {}'.format(self.B, batch_sort))
        else:
            self.y_list_files = None
            wlog('Monolingual: batch size {}, Sort in batch? {}'.format(self.B, batch_sort))

        self.n_batches = int(math.ceil(self.n_sent / self.B))
        self.prefix = prefix    # the prefix of data file, such as 'nist02' or 'nist03'

    def __len__(self):

        return self.n_batches

    def handle_batch(self, batch, right_align=False, bow=False):

        #multi_files = True if isinstance(batch[0], list) else False
        #if multi_files is True:
        # [sent_0:[ref0, ref1, ...], sent_1:[ref0, ref1, ... ], ...]
        # -> [ref_0:[sent_0, sent_1, ...], ref_1:[sent_0, sent_1, ... ], ...]
        batch_for_files = [[one_sent_refs[ref_idx] for one_sent_refs in batch] \
                for ref_idx in range(len(batch[0]))]
        #else:
            # [src0, src1, ...] -> [ [src0, src1, ...] ]
        #    batch_for_files = [batch]

        pad_batch_for_files, lens_for_files = [], []
        pad_batch_bow_for_files = [] if bow is True else None
        for batch in batch_for_files:   # a batch for one source/target file
            #lens = [ts.size(0) for ts in batch]
            #lens = [len(sent_list) for sent_list in batch]
            lens = []
            if bow is True: batch_bow_lists = []
            for sent_list in batch:
                lens.append(len(sent_list))
                if bow is True:
                    bag_of_words = list(set(sent_list))
                    if BOS in bag_of_words: bag_of_words.remove(BOS)    # do not include BOS in bow
                    batch_bow_lists.append(bag_of_words)
            self.this_batch_size = len(batch)
            max_len_batch = max(lens)
            if bow is True:
                bow_lens = [len(bow_list) for bow_list in batch_bow_lists]
                max_bow_len_batch = max(bow_lens)

            # (B, L)
            #pad_batch = tc.Tensor(self.this_batch_size, max_len_batch).long()
            #pad_batch.fill_(PAD)
            pad_batch = [[] for _ in range(self.this_batch_size)]
            if bow is True: pad_batch_bow = [[] for _ in range(self.this_batch_size)]
            for idx in range(self.this_batch_size):
                length = lens[idx]
                #offset = max_len_batch - length if right_align else 0
                # modify Tensor pad_batch
                #pad_batch[idx].narrow(0, offset, length).copy_(batch[idx])
                pad_batch[idx] = batch[idx] + [0] * (max_len_batch - length)
                if bow is True:
                    pad_batch_bow[idx] = batch_bow_lists[idx] + [0] * (max_bow_len_batch - bow_lens[idx])

            pad_batch_for_files.append(pad_batch)
            lens_for_files.append(lens)
            if pad_batch_bow_for_files is not None: pad_batch_bow_for_files.append(pad_batch_bow)

        return pad_batch_for_files, lens_for_files, pad_batch_bow_for_files

    def __getitem__(self, idx):

        assert idx < self.n_batches, 'idx:{} >= number of batches:{}'.format(idx, self.n_batches)

        src_batch = self.x_list[idx * self.B : (idx + 1) * self.B]

        srcs, slens, _ = self.handle_batch(src_batch)
        assert len(srcs) == 1, 'Requires only one in source side.'
        srcs, slens = srcs[0], slens[0]

        if self.y_list_files is not None:
            # [sent_0:[ref0, ref1, ...], sent_1:[ref0, ref1, ... ], ...]
            trg_batch = self.y_list_files[idx * self.B : (idx + 1) * self.B]
            trgs_for_files, tlens_for_files, trg_bows_for_files = self.handle_batch(
                trg_batch, bow=self.bow)
            # -> [ref_0:[sent_0, sent_1, ...], ref_1:[sent_0, sent_1, ... ], ...]
            # trg_bows_for_files -> [ref_0:[bow_0, bow_1, ...], ref_1:[bow_0, bow_1, ... ], ...]

        # sort the source and target sentence
        idxs = range(self.this_batch_size)

        if self.batch_sort is True:
            if self.y_list_files is None:
                zipb = zip(idxs, srcs, slens)
                idxs, srcs, slens = zip(*sorted(zipb, key=lambda x: x[-1]))
            else:
                # max length in different refs may differ, so can not tc.stack
                if trg_bows_for_files is None:
                    zipb = zip(idxs, srcs, zip(*trgs_for_files), slens)
                    idxs, srcs, trgs, slens = zip(*sorted(zipb, key=lambda x: x[-1]))
                    #trgs_for_files = [tc.stack(ref) for ref in zip(*list(trgs))]
                else:
                    zipb = zip(idxs, srcs, zip(*trgs_for_files), zip(*trg_bows_for_files), slens)
                    idxs, srcs, trgs, trg_bows, slens = zip(*sorted(zipb, key=lambda x: x[-1]))
                    trg_bows_for_files = [tc.LongTensor(ref_bow) for ref_bow in zip(*list(trg_bows))]
                trgs_for_files = [tc.LongTensor(ref) for ref in zip(*list(trgs))]

        lengths = tc.IntTensor(slens).view(1, -1)   # (1, batch_size)
        lengths = tc.tensor(lengths)

        def tuple2Tenser(x):
            if x is None: return x
            # (batch_size, max_len_batch)
            if isinstance(x, tuple) or isinstance(x, list): x = tc.LongTensor(x)
            if self.gpu_id: x = x.cuda()    # push into GPU
            return x

        tsrcs = tuple2Tenser(srcs)
        src_mask = tsrcs.ne(0).float()

        if self.y_list_files is not None:

            ttrgs_for_files = [tuple2Tenser(trgs) for trgs in trgs_for_files]
            trg_mask_for_files = [ttrgs.ne(0).float() for ttrgs in ttrgs_for_files]
            if trg_bows_for_files is not None:
                ttrg_bows_for_files = [tuple2Tenser(trg_bows) for trg_bows in trg_bows_for_files]
                ttrg_bows_mask_for_files = [ttrg_bows.ne(0).float() for ttrg_bows in ttrg_bows_for_files]
            else: ttrg_bows_for_files, ttrg_bows_mask_for_files = None, None

            '''
                [list] idxs: sorted idx by ascending order of source lengths in one batch
                [tensor] tsrcs: padded source batch, tensor(batch_size, max_len_batch)
                [list] ttrgs_for_files: list of tensors (padded target batch),
                            [tensor(batch_size, max_len_batch), ..., ]
                            each item in this list for one target reference file one batch
                [intTensor] lengths: sorted source lengths by ascending order, (1, batch_size)
                [tensor] src_mask: 0/1 tensor(0 for padding) (batch_size, max_len_batch)
                [list] trg_mask_for_files: list of 0/1 Variables (0 for padding)
                            [tensor(batch_size, max_len_batch), ..., ]
                            each item in this list for one target reference file one batch
            '''
            return idxs, tsrcs, ttrgs_for_files, ttrg_bows_for_files, lengths, \
                    src_mask, trg_mask_for_files, ttrg_bows_mask_for_files

        else:

            return idxs, tsrcs, lengths, src_mask


    def shuffle(self):

        data = list(zip(self.x_list, self.y_list_files))
        x_tuple, y_tuple = zip(*[data[i] for i in tc.randperm(len(data))])
        self.x_list, self.y_list_files = list(x_tuple), list(y_tuple)

        slens = [len(self.x_list[k]) for k in range(self.n_sent)]
        self.x_list, self.y_list_files = sort_batches(self.x_list, self.y_list_files,
                                                      slens, wargs.batch_size,
                                                      wargs.sort_k_batches)



