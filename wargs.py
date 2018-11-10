# Maximal sequence length in training data
#max_seq_len = 10000000
max_seq_len = 128

# 'cnn', 'att', 'sru', 'gru', 'lstm', 'tgru'
''' encoder '''
encoder_type = 'att'
d_src_emb = 512     # size of source word embedding
n_enc_layers = 2    # layers number
d_enc_hid = 512     # hidden size in rnn

''' decoder '''
decoder_type = 'att'
d_trg_emb = 512     # size of target word embedding
n_dec_layers = 2    # layers number
d_dec_hid = 512     # hidden size in rnn

''' transformer '''
d_model = 512       # n_head * d_v, size of alignment
d_ff_filter = 512  # hidden size of the second layer of PositionwiseFeedForward
n_head = 8          # the number of head for MultiHeadedAttention
att_dropout = 0.3
residual_dropout = 0.3
relu_dropout = 0.3

# dropout for tgru
input_dropout = 0.5
rnn_dropout = 0.3
output_dropout = 0.5

proj_share_weight = True
embs_share_weight = False
position_encoding = True if (encoder_type in ('att','tgru') and decoder_type in ('att','tgru')) else False

''' directory to save model, validation output and test output '''
dir_model = 'wmodel'
dir_valid = 'wvalid'
dir_tests = 'wtests'

''' training data '''
dir_data = 'data/'
train_prefix = 'train'
train_src_suffix = 'src'
train_trg_suffix = 'trg'

''' validation data '''
val_shuffle = True
dev_max_seq_len = 10000000

''' vocabulary '''
word_piece = False
n_src_vcb_plan = 30000
n_trg_vcb_plan = 30000
src_vcb = dir_data + 'src.vcb'
trg_vcb = dir_data + 'trg.vcb'

inputs_data = dir_data + 'inputs.pt'

cased = False
with_bpe = False
with_postproc = False
use_multi_bleu = True

''' training '''
epoch_shuffle_train = True
epoch_shuffle_batch = True
batch_type = 'sents'    # 'sents' or 'tokens', sents is default, tokens will do dynamic batching
sort_k_batches = 20
save_one_model = True
start_epoch = 1
model_prefix = dir_model + '/model'
best_model = dir_valid + '/best.model.pt' if dir_valid else 'best.model.pt'
label_smoothing = 0.1
trg_bow = True
emb_loss = False
bow_loss = False
trunc_size = 0   # truncated bptt
grad_accum_count = 1   # accumulate gradient for batch_size * accum_count batches (Transformer)
snip_size = 20
normalization = 'tokens'     # 'sents' or 'tokens', normalization method of the gradient
max_grad_norm = 5 # the norm of the gradient vector exceeds this, renormalize it to max_grad_norm

''' whether use pretrained model '''
pre_train = None
#pre_train = best_model
fix_pre_params = False
''' start decaying every epoch after and including this epoch '''
start_decay_from = None
learning_rate_decay = 0.5
last_valid_bleu = 0.

''' display settings '''
small = True
display_freq = 10 if small else 1000
look_freq = 100 if small else 5000
n_look = 5
fix_looking = False

''' evaluate settings '''
eval_small = False
epoch_eval = True
src_char = False
char_bleu = False
eval_valid_from = 500 if eval_small else 100000
eval_valid_freq = 100 if eval_small else 20000

''' decoder settings '''
search_mode = 1
with_batch = 1
ori_search = 0
beam_size = 4
vocab_norm = 1  # softmax
len_norm = 2    # 0: no noraml, 1: length normal, 2: alpha-beta
with_mv = 0
merge_way = 'Y'
avg_att = 0
m_threshold = 100.
ngram = 3
alpha_len_norm = 0.6
beta_cover_penalty = 0.

copy_attn = False
file_tran_dir = 'wexp-gpu-nist03'
laynorm = False
segments = False
seg_val_tst_dir = 'orule_1.7'

''' relation network: convolutional layer '''
fltr_windows = [1, 3]
d_fltr_feats = [128, 256]
d_mlp = 256

print_att = True

''' Scheduled Sampling of Samy bengio's paper '''
greed_sampling = False
greed_gumbel_noise = 0.5     # None: w/o noise
bleu_sampling = False
bleu_gumbel_noise = 0.5     # None: w/o noise
ss_type = None     # 1: linear decay, 2: exponential decay, 3: inverse sigmoid decay
ss_eps_begin = 1.   # set None for no scheduled sampling
ss_eps_end = 1.
ss_decay_rate = (ss_eps_begin - ss_eps_end) / 10.
ss_k = 12.     # k < 1 for exponential decay, k >= 1 for inverse sigmoid decay

''' self-normalization settings '''
self_norm_alpha = None  # None or 0.5
nonlocal_mode = 'dot'  # gaussian, dot, embeddedGaussian
# car nmt
#sampling = 'truncation'     # truncation, length_limit, gumbeling
sampling = 'length_limit'     # truncation, length_limit, gumbeling
gpu_id = [0]
#gpu_id = None
n_co_models = len(gpu_id)
s_step_decay = 500 * n_co_models
e_step_decay = 32000 * n_co_models

# 'toy', 'zh-en', 'en-de', 'de-en', 'uy-zh'
dataset = 'toy'
if dataset == 'toy':
    val_tst_dir = './data/'
    val_prefix = 'devset1_2.lc'
    val_src_suffix = 'zh'
    val_ref_suffix = 'en'
    tests_prefix = ['devset3.lc']
    batch_size = 40
    max_epochs = 50
    ''' optimizer settings '''
    opt_mode = 'adam'       # 'adadelta', 'adam' or 'sgd'
    lr_update_way = 't2t'  # 't2t' or 'chen'
    param_init_D = 'X'      # 'U': uniform , 'X': xavier, 'N': normal
    learning_rate = 1.    # 1.0, 0.001, 0.01
    rho = 0.95
    beta_1 = 0.9
    beta_2 = 0.998
    warmup_steps = 2000
elif dataset == 'de-en':
    #val_tst_dir = '/home5/wen/2.data/iwslt14-de-en/'
    val_tst_dir = '/home/wen/3.corpus/mt/iwslt14-de-en/'
    val_prefix = 'valid.de-en'
    val_src_suffix = 'de'
    val_ref_suffix = 'en'
    tests_prefix = ['test.de-en']
    #n_src_vcb_plan = 32009
    #n_trg_vcb_plan = 22822
elif dataset == 'zh-en':
    #val_tst_dir = '/home/wen/3.corpus/mt/nist_data_stanseg/'
    val_tst_dir = '/home/wen/3.corpus/mt/mfd_1.25M/nist_test_new/'
    #val_tst_dir = '/home5/wen/2.data/mt/mfd_1.25M/nist_test_new/'
    val_prefix = 'mt06_u8'
    #dev_prefix = 'nist02'
    val_src_suffix = 'src.BPE'
    val_ref_suffix = 'trg.tok.sb'
    n_src_vcb_plan = 50000
    n_trg_vcb_plan = 50000
    tests_prefix = ['mt02_u8', 'mt03_u8', 'mt04_u8', 'mt05_u8', 'mt08_u8']
    batch_size = 128
    max_epochs = 10
    with_bpe = True
elif dataset == 'uy-zh':
    #val_tst_dir = '/home5/wen/2.data/mt/uy_zh_300w/devtst/'
    val_tst_dir = '/home/wen/3.corpus/mt/uy_zh_300w/devtst/'
    val_prefix = 'dev700'
    #dev_prefix = 'dev700'
    val_src_suffix = '8kbpe.src'
    val_src_suffix = 'uy.src'
    #val_src_suffix = 'uy.32kbpe.src'
    tests_prefix = ['tst861']
elif dataset == 'en-de':
    val_tst_dir = '/home4/wen/3.corpus/wmt14-ende/devtst/'
    val_prefix = 'newstest1213'
    use_multi_bleu = True
    val_src_suffix = 'en.tc.37kbpe'
    val_ref_suffix = 'de.tc' if use_multi_bleu is True else 'ori.de'
    tests_prefix = ['newstest2014']
    n_src_vcb_plan = 50000
    n_trg_vcb_plan = 50000
    batch_size = 128
    sort_k_batches = 32
    with_bpe = True
    cased = True    # False: Case-insensitive BLEU  True: Case-sensitive BLEU


