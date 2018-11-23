# Maximal sequence length in training data
max_seq_len = 128
worse_counter = 0

# 'cnn', 'att', 'sru', 'gru', 'lstm', 'tgru'
''' encoder and decoder '''
encoder_type, decoder_type = 'att', 'att'
d_src_emb, d_trg_emb = 512, 512     # size of source and target word embedding
n_enc_layers, n_dec_layers = 2, 2    # layers number of encoder and decoder
d_enc_hid, d_dec_hid = 512, 512     # hidden size of rnn in encoder and decoder

''' transformer '''
d_model = 512       # n_head * d_v, size of alignment
d_ff_filter = 2048  # hidden size of the second layer of PositionwiseFeedForward
n_head = 8          # the number of head for MultiHeadedAttention

# dropout for tgru
input_dropout, rnn_dropout, output_dropout = 0.5, 0.3, 0.5

proj_share_weight, embs_share_weight = False, False
position_encoding = True if (encoder_type in ('att','tgru') and decoder_type in ('att','tgru')) else False

''' directory to save model, validation output and test output '''
dir_model, dir_valid, dir_tests = 'wmodel', 'wvalid', 'wtests'

''' training data '''
dir_data = 'data/'
train_prefix, train_src_suffix, train_trg_suffix = 'train', 'src', 'trg'

''' validation data '''
dev_max_seq_len = 10000000

''' vocabulary '''
n_src_vcb_plan, n_trg_vcb_plan = 30000, 30000
src_vcb, trg_vcb = dir_data + 'src.vcb', dir_data + 'trg.vcb'

inputs_data = dir_data + 'inputs.pt'

cased, with_bpe, with_postproc, use_multi_bleu = False, False, False, True

''' training '''
epoch_shuffle_train, epoch_shuffle_batch = True, False
sort_k_batches = 100      # 0 for all sort, 1 for no sort
save_one_model = True
start_epoch = 1
trg_bow, emb_loss, bow_loss = True, False, False
trunc_size = 0   # truncated bptt
grad_accum_count = 1   # accumulate gradient for batch_size * accum_count batches (Transformer)
snip_size = 20
normalization = 'tokens'     # 'sents' or 'tokens', normalization method of the gradient
max_grad_norm = 5. # the norm of the gradient vector exceeds this, renormalize it to max_grad_norm
label_smoothing = 0.1
model_prefix = dir_model + '/model'
best_model = dir_valid + '/best.model.pt' if dir_valid else 'best.model.pt'

''' whether use pretrained model '''
pre_train = None
#pre_train = best_model
fix_pre_params = False
''' start decaying every epoch after and including this epoch '''
start_decay_from = None
learning_rate_decay = 0.5
last_valid_bleu = 0.

''' display settings '''
n_look, fix_looking, small = 5, False, False

''' evaluate settings '''
epoch_eval, src_char, char_bleu, eval_small = False, False, False, False
eval_valid_from = 500 if eval_small else 50000
eval_valid_freq = 100 if eval_small else 5000

''' decoder settings '''
search_mode = 1
with_batch, ori_search, vocab_norm = 1, 0, 1
len_norm = 2    # 0: no noraml, 1: length normal, 2: alpha-beta
with_mv, avg_att, m_threshold, ngram = 0, 0, 100., 3
merge_way = 'Y'
beam_size, alpha_len_norm, beta_cover_penalty = 4, 0.6, 0.

copy_attn = False
file_tran_dir = 'wexp-gpu-nist03'
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
n_co_models = 1
s_step_decay = 4000 * n_co_models
e_step_decay = 32000 * n_co_models

opt_mode = 'adam'       # 'adadelta', 'adam' or 'sgd'
beta_1, beta_2, u_gain, adam_epsilon = 0.9, 0.98, 0.08, 1e-9

# 'toy', 'zhen', 'ende', 'deen', 'uyzh'
dataset = 'toy'
model_config = 't2t_tiny'
batch_type = 'token'    # 'sents' or 'tokens', sents is default, tokens will do dynamic batching
batch_size = 40 if batch_type == 'sents' else 2048
if model_config == 't2t_tiny':
    lr_update_way = 't2t'  # 't2t' or 'chen'
    param_init_D = 'X'      # 'U': uniform , 'X': xavier, 'N': normal
    learning_rate, warmup_steps = 1., 300
    input_dropout, att_dropout, relu_dropout, residual_dropout = 0.5, 0.1, 0.1, 0.1
    d_ff_filter, n_head = 512, 8
    small, eval_valid_from, eval_valid_freq = True, 5000, 100
    epoch_eval, max_grad_norm = True, 0.
if model_config == 't2t_base':
    lr_update_way = 't2t'  # 't2t' or 'chen'
    param_init_D = 'X'      # 'U': uniform , 'X': xavier, 'N': normal
    learning_rate, beta_2, warmup_steps = 0.2, 0.997, 8000
    n_enc_layers, n_dec_layers = 6, 6
    input_dropout, att_dropout, relu_dropout, residual_dropout = 0.1, 0.1, 0.1, 0.1
    max_grad_norm = 0.
if model_config == 't2t_big':
    lr_update_way = 't2t'  # 't2t' or 'chen'
    param_init_D = 'X'      # 'U': uniform , 'X': xavier, 'N': normal
    learning_rate, beta_2, warmup_steps = 0.2, 0.997, 8000
    n_enc_layers, n_dec_layers = 6, 6
    d_src_emb, d_trg_emb, d_dec_hid, d_model, d_ff_filter, n_head = 1024, 1024, 1024, 1024, 4096, 16
    input_dropout, att_dropout, relu_dropout, residual_dropout = 0.3, 0.1, 0.1, 0.3
    snip_size, batch_size = 1, 40
    max_grad_norm = 0.
if model_config == 'tgru_base':
    lr_update_way = 'chen'  # 't2t' or 'chen'
    param_init_D = 'U'      # 'U': uniform , 'X': xavier, 'N': normal
    learning_rate = 0.001    # 1.0, 0.001, 0.01
    beta_2, warmup_steps, adam_epsilon = 0.999, 500, 1e-6
if model_config == 'tgru_big':
    lr_update_way = 'chen'  # 't2t' or 'chen'
    param_init_D = 'U'      # 'U': uniform , 'X': xavier, 'N': normal
    learning_rate = 0.001    # 1.0, 0.001, 0.01
    beta_2, warmup_steps, adam_epsilon = 0.999, 500, 1e-6
    d_src_emb, d_trg_emb, d_enc_hid, d_dec_hid, n_head = 1024, 1024, 1024, 1024, 16
if model_config == 'gru_base':
    lr_update_way = 'chen'  # 't2t' or 'chen'
    param_init_D = 'U'      # 'U': uniform , 'X': xavier, 'N': normal
    learning_rate = 0.002    # 1.0, 0.001, 0.01
    beta_2, warmup_steps, adam_epsilon = 0.999, 8000, 1e-6
    s_step_decay, e_step_decay = 8000, 128000
    #d_src_emb, d_trg_emb, d_enc_hid, d_dec_hid = 1024, 1024, 1024, 1024
    snip_size, n_enc_layers = 10, 6

if dataset == 'toy':
    val_tst_dir = './data/'
    val_src_suffix, val_ref_suffix = 'zh', 'en'
    val_prefix, tests_prefix = 'devset1_2.lc', ['devset3.lc']
    max_epochs = 50
elif dataset == 'deen':
    #val_tst_dir = '/home5/wen/2.data/iwslt14-de-en/'
    val_tst_dir = '/home/wen/3.corpus/mt/iwslt14-de-en/'
    val_src_suffix, val_ref_suffix = 'de', 'en'
    val_prefix, tests_prefix = 'valid.de-en', ['test.de-en']
    #n_src_vcb_plan, n_trg_vcb_plan = 32009, 22822
elif dataset == 'zhen':
    #val_tst_dir = '/home/wen/3.corpus/mt/nist_data_stanseg/'
    #val_tst_dir = '/home/wen/3.corpus/mt/mfd_1.25M/nist_test_new/'
    val_tst_dir = '/home5/wen/2.data/mt/mfd_1.25M/nist_test_new/'
    #dev_prefix = 'nist02'
    val_src_suffix, val_ref_suffix = 'src.BPE', 'trg.tok.sb'
    n_src_vcb_plan, n_trg_vcb_plan = 50000, 50000
    val_prefix, tests_prefix = 'mt06_u8', ['mt02_u8', 'mt03_u8', 'mt04_u8', 'mt05_u8', 'mt08_u8']
    batch_size, max_epochs = 100, 15
    with_bpe = True
elif dataset == 'uyzh':
    #val_tst_dir = '/home5/wen/2.data/mt/uy_zh_300w/devtst/'
    val_tst_dir = '/home/wen/3.corpus/mt/uy_zh_300w/devtst/'
    val_src_suffix, val_src_suffix = '8kbpe.src', 'uy.src'
    val_prefix, tests_prefix = 'dev700', ['tst861']
elif dataset == 'ende':
    val_tst_dir = '/home4/wen/3.corpus/wmt14-ende/devtst/'
    use_multi_bleu = True
    val_src_suffix, val_ref_suffix = 'en.tc.37kbpe', 'de.tc'
    val_prefix, tests_prefix = 'newstest1213', ['newstest2014']
    n_src_vcb_plan, n_trg_vcb_plan = 50000, 50000
    batch_size, sort_k_batches = 128, 32
    with_bpe = True
    cased = True    # False: Case-insensitive BLEU  True: Case-sensitive BLEU

display_freq = 10 if small else 1000
look_freq = 100 if small else 5000


