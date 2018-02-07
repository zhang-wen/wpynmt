dataset = 'L' # S for 40k, M for 1.2M, L for wmt en-de

# Maximal sequence length in training data
#max_seq_len = 10000000
max_seq_len = 80

'''
Embedding layer
'''
# Size of word embedding of source word and target word
src_wemb_size = 512
trg_wemb_size = 512

'''
Encoder layer
'''
# Size of hidden units in encoder
enc_hid_size = 512

'''
Attention layer
'''
# Size of alignment vector
align_size = 512

'''
Decoder layer
'''
# Size of hidden units in decoder
dec_hid_size = 512
# Size of the output vector
out_size = 512
drop_rate = 0.5

# Directory to save model, test output and validation output
dir_model = 'wmodel'
dir_valid = 'wvalid'
dir_tests = 'wtests'

# Validation data
val_shuffle = True
# Training data
train_shuffle = True
batch_size = 80
sort_k_batches = 20

# Data path
dir_data = 'data/'
train_prefix = 'train'
train_src_suffix = 'src'
train_trg_suffix = 'trg'
dev_max_seq_len = 10000000

# Dictionary
word_piece = False
src_dict_size = 30000
trg_dict_size = 30000
src_dict = dir_data + 'src.dict.tcf'
trg_dict = dir_data + 'trg.dict.tcf'

inputs_data = dir_data + 'inputs.pt'

with_bpe = False
with_postproc = False
copy_trg_emb = False
# Training
max_epochs = 20
epoch_shuffle = False
epoch_shuffle_minibatch = 1

small = False
eval_small = False
epoch_eval = False
final_test = False

if dataset == 'S':
    src_wemb_size = 256
    trg_wemb_size = 256
    enc_hid_size = 256
    align_size = 256
    dec_hid_size = 256
    out_size = 256
    val_tst_dir = './data/'
    val_prefix = 'devset1_2.lc'
    dev_prefix = 'devset1_2.lc'
    val_src_suffix = 'zh'
    val_ref_suffix = 'en'
    ref_cnt = 16
    tests_prefix = ['devset3.lc']
    batch_size = 10
    max_epochs = 60
    epoch_eval = True
    small = True
    use_multi_bleu = False
    cased = False
elif dataset == 'M':
    src_wemb_size = 512
    trg_wemb_size = 512
    enc_hid_size = 512
    align_size = 512
    dec_hid_size = 512
    out_size = 512
    val_tst_dir = '/home5/wen/2.data/mt/nist_data_stanseg/'
    #val_tst_dir = '/home/wen/3.corpus/mt/nist_data_stanseg/'
    val_prefix = 'nist02'
    dev_prefix = 'nist02'
    #val_src_suffix = '8kbpe.src'
    val_src_suffix = 'src'
    val_ref_suffix = 'ref.plain_'
    ref_cnt = 4
    tests_prefix = ['nist03', 'nist04', 'nist05', 'nist06', 'nist08', '900']
    with_bpe = True
    with_postproc = True
    use_multi_bleu = False
    cased = False
elif dataset == 'L':
    #src_wemb_size = 500
    #trg_wemb_size = 500
    #enc_hid_size = 1024
    #align_size = 1024
    #dec_hid_size = 1024
    #out_size = 512
    #val_tst_dir = '/home/wen/3.corpus/wmt16/rsennrich/devtst/'
    #val_tst_dir = '/home/wen/3.corpus/wmt14/en-de-Luong/'
    val_tst_dir = '/home/wen/3.corpus/wmt2017/de-en/'
    val_prefix = 'newstest2014'
    #val_prefix = 'newstest2014.tc'
    val_src_suffix = 'en.16kbpe'
    val_ref_suffix = 'ori.de'
    ref_cnt = 1
    tests_prefix = ['newstest2014.2737', 'newstest2015', 'newstest2016', 'newstest2017']
    #tests_prefix = ['newstest2009', 'newstest2010', 'newstest2011', 'newstest2012', 'newstest2014', 'newstest2015', 'newstest2016', 'newstest2017']
    #drop_rate = 0.2
    src_dict_size = 50000
    trg_dict_size = 50000
    with_bpe = True
    use_multi_bleu = False
    cased = True    # False: Case-insensitive BLEU  True: Case-sensitive BLEU
    small = True
    eval_small = True

display_freq = 10 if small else 1000
sampling_freq = 100 if small else 5000
sample_size = 5
if_fixed_sampling = False
eval_valid_from = 500 if eval_small else 100000
eval_valid_freq = 100 if eval_small else 20000

save_one_model = True
start_epoch = 1

model_prefix = dir_model + '/model'
best_model = dir_valid + '/best.model.pt' if dir_valid else 'best.model.pt'
# pretrained model
pre_train = None
#pre_train = best_model
fix_pre_params = False

# decoder hype-parameters
search_mode = 1
with_batch = 1
ori_search = 0
beam_size = 10
vocab_norm = 1  # softmax
len_norm = 1    # 0: no noraml, 1: length normal, 2: alpha-beta
with_mv = 0
merge_way = 'Y'
avg_att = 0
m_threshold = 100.
ngram = 3
alpha_len_norm = 0.6
beta_cover_penalty = 0.

'''
Starting learning rate. If adagrad/adadelta/adam is used, then this is the global learning rate.
Recommended settings: sgd = 1, adagrad = 0.1, adadelta = 1, adam = 0.001
'''
opt_mode = 'adadelta'
learning_rate = 1.0

#opt_mode = 'adam'
#learning_rate = 1e-3

#opt_mode = 'sgd'
#learning_rate = 1.

max_grad_norm = 1.0

# Start decaying every epoch after and including this epoch
start_decay_from = None
learning_rate_decay = 0.5
last_valid_bleu = 0.

snip_size = 1
file_tran_dir = 'wexp-gpu-nist03'
laynorm = False
segments = False
seg_val_tst_dir = 'orule_1.7'

# model
enc_rnn_type = 'sru'    # rnn, gru, lstm, sru
enc_layer_cnt = 4
dec_rnn_type = 'sru'    # rnn, gru, lstm, sru
dec_layer_cnt = 4

# 0: groundhog, 1: rnnsearch, 2: ia, 3: ran, 4: rn, 5: sru, 6: cyknet
model = 1

# convolutional layer
#fltr_windows = [1, 3, 5]   # windows size
#d_fltr_feats = [32, 64, 96]
fltr_windows = [3]
d_fltr_feats = [256]
d_mlp = 256

# generate BTG tree when decoding
dynamic_cyk_decoding = False
print_att = True

# Scheduled Sampling of Samy bengio's paper
bleu_sampling = False
ss_type = 3     # 1: linear decay, 2: exponential decay, 3: inverse sigmoid decay
ss_eps_begin = 1.   # set None for no scheduled sampling
ss_eps_end = 1.
#ss_decay_rate = 0.005
ss_decay_rate = (ss_eps_begin - ss_eps_end) / 10.
ss_k = 12.     # k < 1 for exponential decay, k >= 1 for inverse sigmoid decay

# free parameter for self-normalization
# 0 is equivalent to the standard neural network objective function.
self_norm_alpha = None
nonlocal_mode = 'dot'  # gaussian, dot, embeddedGaussian
# car nmt
#sampling = 'truncation'     # truncation, length_limit, gumbeling
sampling = 'length_limit'     # truncation, length_limit, gumbeling
#tests_prefix = None
#dec_gpu_id = [1]
#dec_gpu_id = None
gpu_id = [1]
#gpu_id = None

