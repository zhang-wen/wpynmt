from __future__ import division
from __future__ import absolute_import

import os
import math
import numpy
import torch as tc

import wargs
from tools.utils import *
from tools.dictionary import Dictionary

import sys
import tools.text_encoder as text_encoder
import tools.tokenizer as tokenizer
from collections import defaultdict

# English tokens
EN_SPACE_TOK = 3
# Chinese tokens
ZH_SPACE_TOK = 16

# 8192
def get_or_generate_vocab(data_file, vocab_file, vocab_size=2**13):
    """Inner implementation for vocab generators.
    Args:
    vocab_filename: relative filename where vocab file is stored
    vocab_file: generated vocabulary file
    vocab_size: target size of the vocabulary constructed by SubwordTextEncoder
    Returns:
    A SubwordTextEncoder vocabulary object.
    """
    if os.path.exists(vocab_file):
      wlog('Load dictionary from file {}'.format(vocab_file))
      vocab = text_encoder.SubwordTextEncoder(vocab_file)
      return vocab

    wlog('Save dictionary file into {}'.format(vocab_file))
    token_counts = defaultdict(int)
    for item in genVcb(data_file):
      for tok in tokenizer.encode(text_encoder.native_to_unicode(item)):
          token_counts[tok] += 1

    vocab = text_encoder.SubwordTextEncoder.build_to_target_size(vocab_size, token_counts, 1, 1e3)
    vocab.store_to_file(vocab_file)

    return vocab

"""Generate a vocabulary from the datasets in sources."""
def genVcb(filepath):
    wlog("Generating vocab from: {}".format(filepath))
    # Use Tokenizer to count the word occurrences.
    with open(filepath, 'r') as train_file:
        file_byte_budget = 1e6
        counter = 0
        countermax = int(os.path.getsize(filepath) / file_byte_budget / 2)
        for line in train_file:
            if counter < countermax:
                counter += 1
            else:
                if file_byte_budget <= 0: break
                line = line.strip()
                file_byte_budget -= len(line)
                counter = 0
                yield line

def extract_vocab(data_file, vocab_file, max_vcb_size=30000):

    if os.path.exists(vocab_file) is True:

        # If vocab file has been exist, we load word dictionary
        wlog('Load dictionary from file {}'.format(vocab_file))
        vocab = Dictionary()
        vocab.load_from_file(vocab_file)

    else:

        vocab = count_vocab(data_file, max_vcb_size)
        vocab.write_into_file(vocab_file)
        wlog('Save dictionary file into {}'.format(vocab_file))

    return vocab

def count_vocab(data_file, max_vcb_size):

    vocab = Dictionary()
    with open(data_file, 'r') as f:
        for sent in f.readlines():
            sent = sent.strip()
            for word in sent.split():
                vocab.add(word)

    # vocab.write_into_file('all.vocab')

    words_cnt = sum(vocab.freq.itervalues())
    new_vocab, new_words_cnt = vocab.keep_vocab_size(max_vcb_size)
    wlog('|Final vocabulary| / |Original vocabulary| = {} / {} = {:4.2f}%'
         .format(new_words_cnt, words_cnt, (new_words_cnt/words_cnt) * 100))

    return new_vocab

def wrap_data(src_data, trg_data, src_vocab, trg_vocab, shuffle=True, sort_data=True, max_seq_len=50):

    srcs, trgs, slens = [], [], []
    srcF = open(src_data, 'r')
    num = len(srcF.readlines())
    srcF.close()
    point_every, number_every = int(math.ceil(num/100)), int(math.ceil(num/10))

    srcF = open(src_data, 'r')
    trgF = open(trg_data, 'r')
    idx, ignore, longer = 0, 0, 0

    while True:

        src_sent = srcF.readline()
        trg_sent = trgF.readline()

        if src_sent == '' and trg_sent == '':
            wlog('\nFinish to read bi-corpus.')
            break

        if numpy.mod(idx + 1, point_every) == 0: wlog('.', False)
        if numpy.mod(idx + 1, number_every) == 0: wlog('{}'.format(idx + 1), False)
        idx += 1

        if src_sent == '' or trg_sent == '':
            wlog('Ignore abnormal blank sentence in line number {}'.format(idx))
            ignore += 1

        src_sent, trg_sent = src_sent.strip(), trg_sent.strip()
        src_words, trg_words = src_sent.split(), trg_sent.split()
        src_len = len(src_words)
        if src_len <= max_seq_len or len(trg_words) <= max_seq_len:

            if wargs.word_piece is False:
                srcs.append(src_vocab.keys2idx(src_words, UNK_WORD))
                trgs.append(trg_vocab.keys2idx(trg_words, UNK_WORD,
                                               bos_word=BOS_WORD,
                                               eos_word=EOS_WORD))
            else:
                src_wids = src_vocab.encode(src_sent)
                trg_wids = trg_vocab.encode(trg_sent)
                srcs.append(ids2Tensor(src_wids))
                trgs.append(ids2Tensor(trg_wids, bos_id=BOS, eos_id=EOS))

            slens.append(src_len)
        else:
            longer += 1

    srcF.close()
    trgF.close()

    train_size = len(srcs)
    assert train_size == idx - ignore - longer, 'Wrong .. '
    wlog('Sentence-pairs count: {}(total) - {}(ignore) - {}(longer) = {}'.format(
        idx, ignore, longer, idx - ignore - longer))

    if shuffle is True:

        rand_idxs = tc.randperm(train_size).tolist()
        srcs = [srcs[k] for k in rand_idxs]
        trgs = [trgs[k] for k in rand_idxs]
        slens = [slens[k] for k in rand_idxs]

    final_srcs, final_trgs = srcs, trgs

    if sort_data is True:

        final_srcs, final_trgs = [], []

        if wargs.sort_k_batches == 0:
            wlog('Sorting the whole training data by ascending order of source length ... ', False)
            # sort the whole training data by ascending order of source length
            _, sorted_idx = tc.sort(tc.IntTensor(slens))
            final_srcs = [srcs[k] for k in sorted_idx]
            final_trgs = [trgs[k] for k in sorted_idx]
        else:
            wlog('Sorting for each {} batches ... '.format(wargs.sort_k_batches), False)

            k_batch = wargs.batch_size * wargs.sort_k_batches
            number = int(math.ceil(train_size / k_batch))

            for start in range(number):
                bsrcs = srcs[start * k_batch : (start + 1) * k_batch]
                btrgs = trgs[start * k_batch : (start + 1) * k_batch]
                bslens = slens[start * k_batch : (start + 1) * k_batch]
                _, sorted_idx = tc.sort(tc.IntTensor(bslens))
                final_srcs += [bsrcs[k] for k in sorted_idx]
                final_trgs += [btrgs[k] for k in sorted_idx]

    wlog('Done.')

    return final_srcs, final_trgs

def val_wrap_data(src_data, src_vocab):

    srcs, slens = [], []
    srcF = open(src_data, 'r')
    idx = 0

    while True:

        src_sent = srcF.readline()
        if src_sent == '':
            wlog('Finish to read ... ', False)
            break
        idx += 1

        if src_sent == '':
            wlog('Error. Ignore abnormal blank sentence in line number {}'.format(idx))
            sys.exit(0)

        src_sent = src_sent.strip()
        src_words = src_sent.split()
        src_len = len(src_words)
        if wargs.word_piece is False:
            srcs.append(src_vocab.keys2idx(src_words, UNK_WORD))
        else:
            src_wids = src_vocab.encode(src_sent)
            srcs.append(ids2Tensor(src_wids))

        slens.append(src_len)

    srcF.close()
    wlog('Done. {} count: {}'.format(src_data, idx))

    return srcs, slens


if __name__ == "__main__":

    wlog('\nPreparing source vocabulary ... ')
    src_vocab = extract_vocab(wargs.train_src, wargs.src_dict, wargs.src_dict_size)

    wlog('\nPreparing target vocabulary ... ')
    trg_vocab = extract_vocab(wargs.train_trg, wargs.trg_dict, wargs.trg_dict_size)

    vocabs = {}
    vocabs['src'], vocabs['trg'] = src_vocab, trg_vocab

    wlog('\nPreparing training set ... ')
    trains = {}
    train_src, train_trg = wrap_data(wargs.train_src, wargs.train_trg, src_vocab, trg_vocab)
    '''
    list [torch.LongTensor (sentence), torch.LongTensor, torch.LongTensor, ...]
    no padding
    '''
    trains['src'], trains['trg'] = train_src, train_trg

    wlog('\nPreparing dev set for tuning ... ')
    devs = {}
    dev_src = wargs.val_tst_dir + wargs.val_prefix + '.src'
    dev_trg = wargs.val_tst_dir + wargs.val_prefix + '.ref0'
    dev_src, dev_trg = wrap_data(dev_src, dev_trg, src_vocab, trg_vocab)
    devs['src'], devs['trg'] = dev_src, dev_trg

    wlog('\nPreparing validation set ... ')
    valids = {}
    valid_src, valid_src_lens = val_wrap_data(
        wargs.val_tst_dir + wargs.val_prefix + '.src', src_vocab)
    valids['src'], valids['len'] = valid_src, valid_src_lens

    inputs = {}
    inputs['vocab'] = vocabs
    inputs['train'] = trains
    inputs['valid'] = valids
    inputs['dev'] = devs

    wlog('\nPreparing test set ... ')
    if wargs.tests_prefix:
        tests = {}
        for prefix in wargs.tests_prefix:
            test_src, _ = val_wrap_data(
                wargs.val_tst_dir + prefix + '.src', src_vocab)
            tests[prefix] = test_src
        inputs['tests'] = tests

    wlog('Saving data to {} ... '.format(wargs.inputs_data), False)
    tc.save(inputs, wargs.inputs_data)
    wlog('Done')












