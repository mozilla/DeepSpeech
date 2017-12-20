#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import os
import sys
from util.gpu import get_available_gpus

log_level_index = sys.argv.index('--log_level') + 1 if '--log_level' in sys.argv else 0
os.environ['TF_CPP_MIN_LOG_LEVEL'] = sys.argv[log_level_index] if log_level_index > 0 and log_level_index < len(sys.argv) else '3'

# Determining memory state of each GPU before anything is loaded
memory_limits = [gpu.memory_limit for gpu in get_available_gpus()]

import shutil
import tensorflow as tf
import numpy as np
import inspect
import multiprocessing

from six.moves import zip, range, filter
from tensorflow.contrib.session_bundle import exporter
from tensorflow.python.tools import freeze_graph
from threading import Thread, Lock, Event
from xdg import BaseDirectory as xdg
from util.feeding import DataSet, ModelFeeder
from util.persistence import CheckpointManager
from util.text import sparse_tensor_value_to_texts, wer, Alphabet
from util.messaging import ClusterMessagingClient
from util.log import set_log_levels, Logger

# Importer
# ========

tf.app.flags.DEFINE_integer ('threads_per_set',  0,           'concurrent sample loader threads per data set (train, dev, test) - default (0) is equal to the number of CPU cores (but at least 2)')
tf.app.flags.DEFINE_integer ('loader_buffer',    0,           'number of samples in the buffer that is used to pick batches from - default (0) is GPU memory of the biggest GPU divided by 10 million (but at least 100)')
tf.app.flags.DEFINE_integer ('queue_capacity',   100,         'capacity of feeding queues (number of samples) - defaults to 100')

# Files

tf.app.flags.DEFINE_string  ('train_files',      '',          'comma separated list of files specifying the dataset used for training. multiple files will get merged')
tf.app.flags.DEFINE_string  ('dev_files',        '',          'comma separated list of files specifying the dataset used for validation. multiple files will get merged')
tf.app.flags.DEFINE_string  ('test_files',       '',          'comma separated list of files specifying the dataset used for testing. multiple files will get merged')

# Batch sizes

tf.app.flags.DEFINE_integer ('train_batch_size', 0,           'number of elements in a training batch - 0 (default) for dynamic batch size')
tf.app.flags.DEFINE_integer ('dev_batch_size',   0,           'number of elements in a validation batch - 0 (default) for dynamic batch size')
tf.app.flags.DEFINE_integer ('test_batch_size',  0,           'number of elements in a test batch - 0 (default) for dynamic batch size')

# Sample window

tf.app.flags.DEFINE_integer ('limit_train',      0,           'maximum number of elements to use from train set - 0 means no limit')
tf.app.flags.DEFINE_integer ('limit_dev',        0,           'maximum number of elements to use from validation set- 0 means no limit')
tf.app.flags.DEFINE_integer ('limit_test',       0,           'maximum number of elements to use from test set- 0 means no limit')
tf.app.flags.DEFINE_integer ('skip_train',       0,           'number of elements to skip from the beginning of the train set')
tf.app.flags.DEFINE_integer ('skip_dev',         0,           'number of elements to skip from the beginning of the validation set')
tf.app.flags.DEFINE_integer ('skip_test',        0,           'number of elements to skip from the beginning of the test set')
tf.app.flags.DEFINE_boolean ('train_ascending',  True,        'process samples in train set in ascending (True) or descending (False) order - default True')
tf.app.flags.DEFINE_boolean ('dev_ascending',    True,        'process samples in validation set in ascending (True) or descending (False) order - default True')
tf.app.flags.DEFINE_boolean ('test_ascending',   True,        'process samples in test set in ascending (True) or descending (False) order - default True')

# Cluster configuration
# =====================

tf.app.flags.DEFINE_string  ('nodes',            '',          'comma separated list of hostname:port pairs of cluster worker nodes')
tf.app.flags.DEFINE_integer ('task_index',       0,           'index of this worker within the cluster - worker with index 0 will be the chief')

# Global Constants
# ================

tf.app.flags.DEFINE_integer ('gpu_allocation',   100,         'how much GPU memory should be allocated in percent')
tf.app.flags.DEFINE_integer ('cpu_memory',       4000000000,  'how much CPU memory in bytes should be used during CPU-only training (defaults to 4000000000)')

tf.app.flags.DEFINE_boolean ('train',            True,        'wether to train the network')
tf.app.flags.DEFINE_boolean ('test',             True,        'wether to test the network')
tf.app.flags.DEFINE_integer ('epoch',            75,          'target epoch to train - if negative, the absolute number of additional epochs will be trained')

tf.app.flags.DEFINE_boolean ('use_warpctc',      False,       'wether to use GPU bound Warp-CTC')

tf.app.flags.DEFINE_float   ('dropout_rate',     0.05,        'dropout rate for feedforward layers')
tf.app.flags.DEFINE_float   ('dropout_rate2',    -1.0,        'dropout rate for layer 2 - defaults to dropout_rate')
tf.app.flags.DEFINE_float   ('dropout_rate3',    -1.0,        'dropout rate for layer 3 - defaults to dropout_rate')
tf.app.flags.DEFINE_float   ('dropout_rate4',    0.0,         'dropout rate for layer 4 - defaults to 0.0')
tf.app.flags.DEFINE_float   ('dropout_rate5',    0.0,         'dropout rate for layer 5 - defaults to 0.0')
tf.app.flags.DEFINE_float   ('dropout_rate6',    -1.0,        'dropout rate for layer 6 - defaults to dropout_rate')

tf.app.flags.DEFINE_float   ('relu_clip',        20.0,        'ReLU clipping value for non-recurrant layers')

# Adam optimizer (http://arxiv.org/abs/1412.6980) parameters

tf.app.flags.DEFINE_float   ('beta1',            0.9,         'beta 1 parameter of Adam optimizer')
tf.app.flags.DEFINE_float   ('beta2',            0.999,       'beta 2 parameter of Adam optimizer')
tf.app.flags.DEFINE_float   ('epsilon',          1e-8,        'epsilon parameter of Adam optimizer')
tf.app.flags.DEFINE_float   ('learning_rate',    0.001,       'learning rate of Adam optimizer')

# Step widths

tf.app.flags.DEFINE_integer ('display_step',     0,           'number of epochs we cycle through before displaying detailed progress - 0 means no progress display')
tf.app.flags.DEFINE_integer ('validation_step',  0,           'number of epochs we cycle through before validating the model - a detailed progress report is dependent on "--display_step" - 0 means no validation steps')

# Checkpointing

tf.app.flags.DEFINE_string  ('checkpoint_dir',   '',          'directory in which checkpoints are stored - defaults to directory "deepspeech/checkpoints" within user\'s data home specified by the XDG Base Directory Specification')
tf.app.flags.DEFINE_string  ('load',             'recent',    'either "recent" to load most recent checkpoint from checkpoint_dir, "best-dev" to load lowest loss epoch checkpoint, "last-epoch" to load the last epoch checkpoint or a checkpoint filename to load - defaults to "recent"')
tf.app.flags.DEFINE_integer ('inter_secs',       600,         'time interval for saving intermediate checkpoints in seconds (0 for turning off intermediate checkpoints) - defaults to 600')
tf.app.flags.DEFINE_integer ('keep_n_epochs',    5,           'number of epoch checkpoint files to keep (0 for not writing epoch checkpoints) - default value is 5')
tf.app.flags.DEFINE_integer ('keep_n_inters',    3,           'number of intermediate checkpoint files to keep (0 for not writing intermediate checkpoints) - default value is 3')

# Exporting

tf.app.flags.DEFINE_string  ('export_dir',       '',          'directory in which exported models are stored - if omitted, the model won\'t get exported')
tf.app.flags.DEFINE_integer ('export_version',   1,           'version number of the exported model')
tf.app.flags.DEFINE_boolean ('remove_export',    False,       'wether to remove old exported models')

# Reporting

tf.app.flags.DEFINE_string  ('log_level',        'info',      'comma separated assignments of log levels to modules (e.g. "main=debug,persistence=step") - Modules can be "main", "persistence", "messaging", "feeding". Levels can be "step", "debug", "info", "warn", "error". If only a level is specified, it will will be used as default value.')

tf.app.flags.DEFINE_boolean ('log_placement',    False,       'wether to log device placement of the operators to the console')
tf.app.flags.DEFINE_integer ('report_count',     10,          'number of phrases with lowest WER (best matching) to print out during a WER report')

tf.app.flags.DEFINE_string  ('summary_dir',      '',          'target directory for TensorBoard summaries - defaults to directory "deepspeech/summaries" within user\'s data home specified by the XDG Base Directory Specification')
tf.app.flags.DEFINE_integer ('summary_secs',     0,           'interval in seconds for saving TensorBoard summaries - if 0, no summaries will be written')

# Geometry

tf.app.flags.DEFINE_integer ('n_hidden',         2048,        'layer width to use when initialising layers')

# Initialization

tf.app.flags.DEFINE_integer ('random_seed',      4567,        'default random seed that is used to initialize variables')
tf.app.flags.DEFINE_float   ('default_stddev',   0.046875,    'default standard deviation to use when initialising weights and biases')

# Early Stopping

tf.app.flags.DEFINE_boolean ('early_stop',       True,        'enable early stopping mechanism over validation dataset. Make sure that dev FLAG is enabled for this to work')
tf.app.flags.DEFINE_integer ('es_nsteps',        4,           'number of steps to consider for early stopping. Loss is not stored in the checkpoint so when checkpoint is revived it starts the loss calculation from start at that point')
tf.app.flags.DEFINE_float   ('es_mean_thresh',   0.5,         'mean threshold for loss to determine the condition if early stopping is required')
tf.app.flags.DEFINE_float   ('es_std_thresh',    0.5,         'standard deviation threshold for loss to determine the condition if early stopping is required')

# Decoder

tf.app.flags.DEFINE_string  ('decoder_library_path', 'native_client/libctc_decoder_with_kenlm.so', 'path to the libctc_decoder_with_kenlm.so library containing the decoder implementation.')
tf.app.flags.DEFINE_string  ('alphabet_config_path', 'data/alphabet.txt', 'path to the configuration file specifying the alphabet used by the network. See the comment in data/alphabet.txt for a description of the format.')
tf.app.flags.DEFINE_string  ('lm_binary_path',       'data/lm/lm.binary', 'path to the language model binary file created with KenLM')
tf.app.flags.DEFINE_string  ('lm_trie_path',         'data/lm/trie', 'path to the language model trie file created with native_client/generate_trie')
tf.app.flags.DEFINE_integer ('beam_width',        1024,       'beam width used in the CTC decoder when building candidate transcriptions')
tf.app.flags.DEFINE_float   ('lm_weight',         2.15,        'the alpha hyperparameter of the CTC decoder. Language Model weight.')
tf.app.flags.DEFINE_float   ('word_count_weight', -0.10,        'the beta hyperparameter of the CTC decoder. Word insertion weight (penalty).')
tf.app.flags.DEFINE_float   ('valid_word_count_weight', 1.10,        'Valid word insertion weight. This is used to lessen the word insertion penalty when the inserted word is part of the vocabulary.')

for var in ['b1', 'h1', 'b2', 'h2', 'b3', 'h3', 'b5', 'h5', 'b6', 'h6']:
    tf.app.flags.DEFINE_float('%s_stddev' % var, None, 'standard deviation to use when initialising %s' % var)

# parse command line parameters
FLAGS = tf.app.flags.FLAGS

# set global cross module log level
set_log_levels(FLAGS.log_level)

# create local logger w/o module name
log = Logger(id='main')

def initialize_globals():
    # nodes required for cluster setup
    FLAGS.nodes = list(filter(len, FLAGS.nodes.split(',')))
    if len(FLAGS.nodes) == 0:
        FLAGS.nodes.append('localhost:0') # port 0 lets system pick a free port

    # create a cluster_spec from worker hosts.
    global cluster_spec
    cluster_spec = tf.train.ClusterSpec({ 'worker': FLAGS.nodes })

    # determine, if we are the chief worker
    global is_chief
    is_chief = FLAGS.task_index == 0

    if len(memory_limits) == 0:
        # no compatible GPU -> CPU training with specified memory allocation
        memory_limits.append(FLAGS.cpu_memory)

    # get CSV lists
    FLAGS.train_files = FLAGS.train_files.split(',')
    FLAGS.dev_files =   FLAGS.dev_files.split(',')
    FLAGS.test_files =  FLAGS.test_files.split(',')

    # by default we run as many sample loading threads per set as CPU cores
    # (as there is only one set active at a time)
    cpu_count = multiprocessing.cpu_count()
    if FLAGS.threads_per_set <= 0:
        FLAGS.threads_per_set = max(2, cpu_count)
    log.debug('Number of loader threads per data set (%d CPUs): %d' % (cpu_count, FLAGS.threads_per_set))

    # by default the loader buffer is the million-th part of the biggest GPU's memory in bytes
    if FLAGS.loader_buffer <= 0:
        FLAGS.loader_buffer = max(100, max(memory_limits) // 1000000) if len(memory_limits) > 0 else 100
    log.debug('Number of samples in loader buffer (%d bytes GPU memory): %d' % (max(memory_limits), FLAGS.loader_buffer))

    # set default dropout rates
    if FLAGS.dropout_rate2 < 0:
        FLAGS.dropout_rate2 = FLAGS.dropout_rate
    if FLAGS.dropout_rate3 < 0:
        FLAGS.dropout_rate3 = FLAGS.dropout_rate
    if FLAGS.dropout_rate6 < 0:
        FLAGS.dropout_rate6 = FLAGS.dropout_rate

    global dropout_rates
    dropout_rates = [ FLAGS.dropout_rate,
                      FLAGS.dropout_rate2,
                      FLAGS.dropout_rate3,
                      FLAGS.dropout_rate4,
                      FLAGS.dropout_rate5,
                      FLAGS.dropout_rate6 ]

    global no_dropout
    no_dropout = [ 0.0 ] * 6

    # set default checkpoint dir
    if len(FLAGS.checkpoint_dir) == 0:
        FLAGS.checkpoint_dir = xdg.save_data_path(os.path.join('deepspeech','checkpoints'))

    # set default summary dir
    if len(FLAGS.summary_dir) == 0:
        FLAGS.summary_dir = xdg.save_data_path(os.path.join('deepspeech','summaries'))

    # standard session configuration that'll be used for all new sessions.
    global session_config
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=float(FLAGS.gpu_allocation) / 100.0)
    session_config = tf.ConfigProto(allow_soft_placement=True,
                                    log_device_placement=FLAGS.log_placement,
                                    gpu_options=gpu_options)

    global alphabet
    alphabet = Alphabet(os.path.abspath(FLAGS.alphabet_config_path))

    # Geometric Constants
    # ===================

    # For an explanation of the meaning of the geometric constants, please refer to
    # doc/Geometry.md

    # number of MFCC features
    global n_input
    n_input = 26 # TODO: Determine this programatically from the sample rate

    # number of frames in the context
    global n_context
    n_context = 9 # TODO: Determine the optimal value using a validation data set

    # number of units in hidden layers
    global n_hidden
    n_hidden = FLAGS.n_hidden

    global n_hidden_1
    n_hidden_1 = n_hidden

    global n_hidden_2
    n_hidden_2 = n_hidden

    global n_hidden_5
    n_hidden_5 = n_hidden

    # LSTM cell state dimension
    global n_cell_dim
    n_cell_dim = n_hidden

    # number of units in the third layer, which feeds in to the LSTM
    global n_hidden_3
    n_hidden_3 = 2 * n_cell_dim

    # number of characters in the target language plus one
    global n_character
    n_character = alphabet.size() + 1 # +1 for CTC blank label

    # number of units in the sixth layer
    global n_hidden_6
    n_hidden_6 = n_character

    # assign default values for standard deviation
    for var in ['b1', 'h1', 'b2', 'h2', 'b3', 'h3', 'b5', 'h5', 'b6', 'h6']:
        val = getattr(FLAGS, '%s_stddev' % var)
        if val is None:
            setattr(FLAGS, '%s_stddev' % var, FLAGS.default_stddev)

# Graph Creation
# ==============

def model_variable(name, shape, initializer, trainable=True):
    '''
    Creates a model variable on chief worker's CPU
    '''
    with tf.device('/job:worker/task:0/cpu'):
        return tf.get_variable(name=name, shape=shape, initializer=initializer, trainable=trainable)

def BiRNN(batch_x, seq_length, dropout):
    r'''
    That done, we will define the learned variables, the weights and biases,
    within the method ``BiRNN()`` which also constructs the neural network.
    The variables named ``hn``, where ``n`` is an integer, hold the learned weight variables.
    The variables named ``bn``, where ``n`` is an integer, hold the learned bias variables.
    In particular, the first variable ``h1`` holds the learned weight matrix that
    converts an input vector of dimension ``n_input + 2*n_input*n_context``
    to a vector of dimension ``n_hidden_1``.
    Similarly, the second variable ``h2`` holds the weight matrix converting
    an input vector of dimension ``n_hidden_1`` to one of dimension ``n_hidden_2``.
    The variables ``h3``, ``h5``, and ``h6`` are similar.
    Likewise, the biases, ``b1``, ``b2``..., hold the biases for the various layers.
    '''

    # Input shape: [batch_size, n_steps, n_input + 2*n_input*n_context]
    batch_x_shape = tf.shape(batch_x)

    # Reshaping `batch_x` to a tensor with shape `[n_steps*batch_size, n_input + 2*n_input*n_context]`.
    # This is done to prepare the batch for input into the first layer which expects a tensor of rank `2`.

    # Permute n_steps and batch_size
    batch_x = tf.transpose(batch_x, [1, 0, 2])
    # Reshape to prepare input for first layer
    batch_x = tf.reshape(batch_x, [-1, n_input + 2*n_input*n_context]) # (n_steps*batch_size, n_input + 2*n_input*n_context)

    # The next three blocks will pass `batch_x` through three hidden layers with
    # clipped RELU activation and dropout.

    # 1st layer
    b1 = model_variable('b1', [n_hidden_1], tf.random_normal_initializer(stddev=FLAGS.b1_stddev))
    h1 = model_variable('h1', [n_input + 2*n_input*n_context, n_hidden_1], tf.contrib.layers.xavier_initializer(uniform=False))
    layer_1 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(batch_x, h1), b1)), FLAGS.relu_clip)
    layer_1 = tf.nn.dropout(layer_1, (1.0 - dropout[0]))

    # 2nd layer
    b2 = model_variable('b2', [n_hidden_2], tf.random_normal_initializer(stddev=FLAGS.b2_stddev))
    h2 = model_variable('h2', [n_hidden_1, n_hidden_2], tf.random_normal_initializer(stddev=FLAGS.h2_stddev))
    layer_2 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(layer_1, h2), b2)), FLAGS.relu_clip)
    layer_2 = tf.nn.dropout(layer_2, (1.0 - dropout[1]))

    # 3rd layer
    b3 = model_variable('b3', [n_hidden_3], tf.random_normal_initializer(stddev=FLAGS.b3_stddev))
    h3 = model_variable('h3', [n_hidden_2, n_hidden_3], tf.random_normal_initializer(stddev=FLAGS.h3_stddev))
    layer_3 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(layer_2, h3), b3)), FLAGS.relu_clip)
    layer_3 = tf.nn.dropout(layer_3, (1.0 - dropout[2]))

    # Now we create the forward and backward LSTM units.
    # Both of which have inputs of length `n_cell_dim` and bias `1.0` for the forget gate of the LSTM.

    # Forward direction cell: (if else required for TF 1.0 and 1.1 compat)
    lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(n_cell_dim, forget_bias=1.0, state_is_tuple=True) \
                   if 'reuse' not in inspect.getargspec(tf.contrib.rnn.BasicLSTMCell.__init__).args else \
                   tf.contrib.rnn.BasicLSTMCell(n_cell_dim, forget_bias=1.0, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)
    lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(lstm_fw_cell,
                                                input_keep_prob=1.0 - dropout[3],
                                                output_keep_prob=1.0 - dropout[3],
                                                seed=FLAGS.random_seed)
    # Backward direction cell: (if else required for TF 1.0 and 1.1 compat)
    lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(n_cell_dim, forget_bias=1.0, state_is_tuple=True) \
                   if 'reuse' not in inspect.getargspec(tf.contrib.rnn.BasicLSTMCell.__init__).args else \
                   tf.contrib.rnn.BasicLSTMCell(n_cell_dim, forget_bias=1.0, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)
    lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(lstm_bw_cell,
                                                input_keep_prob=1.0 - dropout[4],
                                                output_keep_prob=1.0 - dropout[4],
                                                seed=FLAGS.random_seed)

    # `layer_3` is now reshaped into `[n_steps, batch_size, 2*n_cell_dim]`,
    # as the LSTM BRNN expects its input to be of shape `[max_time, batch_size, input_size]`.
    layer_3 = tf.reshape(layer_3, [-1, batch_x_shape[0], n_hidden_3])

    # Now we feed `layer_3` into the LSTM BRNN cell and obtain the LSTM BRNN output.
    outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell,
                                                             cell_bw=lstm_bw_cell,
                                                             inputs=layer_3,
                                                             dtype=tf.float32,
                                                             time_major=True,
                                                             sequence_length=seq_length)

    # Reshape outputs from two tensors each of shape [n_steps, batch_size, n_cell_dim]
    # to a single tensor of shape [n_steps*batch_size, 2*n_cell_dim]
    outputs = tf.concat(outputs, 2)
    outputs = tf.reshape(outputs, [-1, 2*n_cell_dim])

    # Now we feed `outputs` to the fifth hidden layer with clipped RELU activation and dropout
    b5 = model_variable('b5', [n_hidden_5], tf.random_normal_initializer(stddev=FLAGS.b5_stddev))
    h5 = model_variable('h5', [(2 * n_cell_dim), n_hidden_5], tf.random_normal_initializer(stddev=FLAGS.h5_stddev))
    layer_5 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(outputs, h5), b5)), FLAGS.relu_clip)
    layer_5 = tf.nn.dropout(layer_5, (1.0 - dropout[5]))

    # Now we apply the weight matrix `h6` and bias `b6` to the output of `layer_5`
    # creating `n_classes` dimensional vectors, the logits.
    b6 = model_variable('b6', [n_hidden_6], tf.random_normal_initializer(stddev=FLAGS.b6_stddev))
    h6 = model_variable('h6', [n_hidden_5, n_hidden_6], tf.contrib.layers.xavier_initializer(uniform=False))
    layer_6 = tf.add(tf.matmul(layer_5, h6), b6)

    # Finally we reshape layer_6 from a tensor of shape [n_steps*batch_size, n_hidden_6]
    # to the slightly more useful shape [n_steps, batch_size, n_hidden_6].
    # Note, that this differs from the input in that it is time-major.
    layer_6 = tf.reshape(layer_6, [-1, batch_x_shape[0], n_hidden_6], name="logits")

    # Output shape: [n_steps, batch_size, n_hidden_6]
    return layer_6

if not os.path.exists(os.path.abspath(FLAGS.decoder_library_path)):
    log.error('ERROR: The decoder library file does not exist. Make sure you have ' \
          'downloaded or built the native client binaries and pass the ' \
          'appropriate path to the binaries in the --decoder_library_path parameter.')

custom_op_module = tf.load_op_library(FLAGS.decoder_library_path)

def decode_with_lm(inputs, sequence_length, beam_width=100,
                   top_paths=1, merge_repeated=True):
  decoded_ixs, decoded_vals, decoded_shapes, log_probabilities = (
      custom_op_module.ctc_beam_search_decoder_with_lm(
          inputs, sequence_length, beam_width=beam_width,
          model_path=FLAGS.lm_binary_path, trie_path=FLAGS.lm_trie_path, alphabet_path=FLAGS.alphabet_config_path,
          lm_weight=FLAGS.lm_weight, word_count_weight=FLAGS.word_count_weight, valid_word_count_weight=FLAGS.valid_word_count_weight,
          top_paths=top_paths, merge_repeated=merge_repeated))

  return (
      [tf.SparseTensor(ix, val, shape) for (ix, val, shape)
       in zip(decoded_ixs, decoded_vals, decoded_shapes)],
      log_probabilities)



# Accuracy and Loss
# =================

# In accord with 'Deep Speech: Scaling up end-to-end speech recognition'
# (http://arxiv.org/abs/1412.5567),
# the loss function used by our network should be the CTC loss function
# (http://www.cs.toronto.edu/~graves/preprint.pdf).
# Conveniently, this loss function is implemented in TensorFlow.
# Thus, we can simply make use of this implementation to define our loss.

def calculate_mean_edit_distance_and_loss(model_feeder, batch):
    r'''
    This routine beam search decodes a mini-batch and calculates the loss and mean edit distance.
    Next to batch size, total and average loss it returns the mean edit distance,
    the decoded result and the batch's original Y.
    '''
    # obtain the next batch of data
    batch.size, batch.x, batch.seq_len, batch.y = model_feeder.next_batch(batch.worker_index, batch.gpu_index)
    # calculate the logits of the batch using BiRNN
    logits = BiRNN(batch.x, tf.to_int64(batch.seq_len), dropout_rates)
    # beam search decode the batch
    batch.decoded, _ = decode_with_lm(logits, batch.seq_len, merge_repeated=False, beam_width=FLAGS.beam_width)
    # compute the CTC loss using either TensorFlow's `ctc_loss` or Baidu's `warp_ctc_loss`.
    if FLAGS.use_warpctc:
        batch.loss = tf.contrib.warpctc.warp_ctc_loss(labels=batch.y, inputs=logits, sequence_length=batch.seq_len)
    else:
        batch.loss = tf.nn.ctc_loss(labels=batch.y, inputs=logits, sequence_length=batch.seq_len)
    # calculate the mean loss
    batch.avg_loss = tf.reduce_mean(batch.loss, 0)
    # compute the edit (Levenshtein) distance
    batch.distance = tf.edit_distance(tf.cast(batch.decoded[0], tf.int32), batch.y)
    # calculate the mean distance
    batch.avg_distance = tf.reduce_mean(batch.distance, 0)
    return batch

# Adam Optimization
# =================

# In constrast to 'Deep Speech: Scaling up end-to-end speech recognition'
# (http://arxiv.org/abs/1412.5567),
# in which 'Nesterov's Accelerated Gradient Descent'
# (www.cs.toronto.edu/~fritz/absps/momentum.pdf) was used,
# we will use the Adam method for optimization (http://arxiv.org/abs/1412.6980),
# because, generally, it requires less fine-tuning.
def create_optimizer(weight):
    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate * weight,
                                       beta1=FLAGS.beta1,
                                       beta2=FLAGS.beta2,
                                       epsilon=FLAGS.epsilon)
    return optimizer

# Towers
# ======

# In order to properly make use of multiple GPU's, one must introduce new abstractions,
# not present when using a single GPU, that facilitate the multi-GPU use case.
# In particular, one must introduce a means to isolate the inference and gradient
# calculations on the various GPU's.
# The abstraction we intoduce for this purpose is called a 'tower'.
# A tower is specified by two properties:
# * **Scope** - A scope, as provided by `tf.name_scope()`,
# is a means to isolate the operations within a tower.
# For example, all operations within 'tower 0' could have their name prefixed with `tower_0/`.
# * **Device** - A hardware device, as provided by `tf.device()`,
# on which all operations within the tower execute.
# For example, all operations of 'tower 0' could execute on the first GPU `tf.device('/gpu:0')`.

def average_gradients(device, gradients, batch_sizes):
    '''
    A routine for computing each variable's average of the gradients obtained from the GPUs.
    Note also that this code acts as a syncronization point as it requires all
    GPUs to be finished with their mini-batch before it can run to completion.
    '''
    # List of average gradients to return to the caller
    average_grads = []
    # Run this on cpu_device to conserve GPU memory
    with tf.device(device):
        # Compute sum of all tower batch sizes
        batch_size_sum = tf.reduce_sum(batch_sizes, 0)
        sample_number = tf.maximum(1, batch_size_sum)
        # Loop over gradient/variable pairs from all towers
        for grad_and_vars in zip(*gradients):
            # Introduce grads to store the gradients for the current variable
            grads = []
            # Loop over the gradients for the current variable
            for gv, batch_size in zip(grad_and_vars, batch_sizes):
                # Weighted gradient - batch size is 0 for dummy sample
                g = gv[0] * tf.to_float(batch_size)
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)
                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)
            # Average over the 'tower' dimension
            grad = tf.concat(grads, 0)
            grad = tf.reduce_sum(grad, 0)
            grad = grad / tf.to_float(sample_number)
            # Create a gradient/variable tuple for the current variable with its average gradient
            grad_and_var = (grad, grad_and_vars[0][1])
            # Add the current tuple to average_grads
            average_grads.append(grad_and_var)
    # Return result to caller
    return average_grads, batch_size_sum

def get_tower_results(model_feeder):
    '''
    With this preliminary step out of the way, we can for each GPU introduce a
    tower for which's batch we calculate

    * The CTC decodings ``decoded``,
    * The (total) loss against the outcome (Y) ``total_loss``,
    * The loss averaged over the whole batch ``avg_loss``,
    * The optimization gradient (computed based on the averaged loss),
    * The Levenshtein distances between the decodings and their transcriptions ``distance``,
    * The mean edit distance of the outcome averaged over the whole batch ``mean_edit_distance``
    '''
    # preparing data bags that represent the tower batches
    class DataBag:
        pass
    tower_batches = []
    for worker in model_feeder.workers:
        for gpu in worker.gpus:
            batch = DataBag()
            batch.worker_index = worker.index
            batch.gpu_index = gpu.index
            tower_batches.append(batch)
    def for_each_tower(callback):
        '''
        Helper routine that executes a provided function for each data bag
        in its appropriate tower context
        '''
        with tf.variable_scope(tf.get_variable_scope()):
            # Loop over available_devices
            for batch in tower_batches:
                wg = (batch.worker_index, batch.gpu_index)
                # Execute operations of tower t on worker w
                device = '/job:worker/task:%d/gpu:%d' % wg
                with tf.device(device):
                    # Create a scope for all operations of tower t
                    with tf.name_scope('tower_%d_%d' % wg):
                        callback(model_feeder, batch)
                        # Allow for variables to be re-used by the next tower
                        tf.get_variable_scope().reuse_variables()
    # building the graph
    for_each_tower(calculate_mean_edit_distance_and_loss)
    # constructing sum of all tower batch sizes
    batch_sizes = [batch.size for batch in tower_batches]
    batch_size_sum = tf.to_float(tf.maximum(1, tf.reduce_sum(batch_sizes)))
    # creating the optimizer by passing it the sum of all batch sizes for weighting
    optimizer = create_optimizer(batch_size_sum)
    def weight_and_compute_gradients(model_feeder, batch):
        batch.gradients = optimizer.compute_gradients(batch.avg_loss)
        batch_size_float = tf.to_float(batch.size)
        batch.avg_loss = batch.avg_loss * batch_size_float
        batch.avg_distance = batch.avg_distance * batch_size_float
    # compute gradients - also get weighted loss and weighted edit-distance
    for_each_tower(weight_and_compute_gradients)
    # building result lists of selected attributes
    labels, decodings, distances, losses, avg_losses, avg_distances = \
        zip(*[(batch.y, batch.decoded, batch.distance, batch.loss, batch.avg_loss, batch.avg_distance) for batch in tower_batches])
    # saving transport bandwidth: first iterate over workers to compute averaged worker gradients...
    worker_averages = []
    for worker in model_feeder.workers:
        # pick worker's gradients and batch sizes
        gradients, batch_sizes = zip(*[(batch.gradients, batch.size) for batch in tower_batches if batch.worker_index == worker.index])
        # average weighted GPU gradients to weighted worker gradients on worker's CPU
        worker_averages.append(average_gradients('/job:worker/task:%d/cpu' % worker.index, gradients, batch_sizes))
    # ... and then average weighted worker gradients to overall step gradients
    # Note: This should avoid transporting all tower gradients to chief node before averaging
    gradients, batch_size = average_gradients('/cpu:0', *zip(*worker_averages))
    # return the optimizer, the results tuple, gradients, the batch size,
    # and weighted means of edit distance and loss
    return optimizer, \
           (labels, decodings, distances, losses), \
           gradients, \
           batch_size, \
           tf.reduce_sum(avg_distances, 0) / batch_size_sum, \
           tf.reduce_sum(avg_losses, 0) / batch_size_sum

def log_variable(variable, gradient=None):
    r'''
    We introduce a function for logging a tensor variable's current state.
    It logs scalar values for the mean, standard deviation, minimum and maximum.
    Furthermore it logs a histogram of its state and (if given) of an optimization gradient.
    '''
    name = variable.name
    mean = tf.reduce_mean(variable)
    tf.summary.scalar(name='%s/mean'   % name, tensor=mean)
    tf.summary.scalar(name='%s/sttdev' % name, tensor=tf.sqrt(tf.reduce_mean(tf.square(variable - mean))))
    tf.summary.scalar(name='%s/max'    % name, tensor=tf.reduce_max(variable))
    tf.summary.scalar(name='%s/min'    % name, tensor=tf.reduce_min(variable))
    tf.summary.histogram(name=name, values=variable)
    if gradient is not None:
        if isinstance(gradient, tf.IndexedSlices):
            grad_values = gradient.values
        else:
            grad_values = gradient
        if grad_values is not None:
            tf.summary.histogram(name='%s/gradients' % name, values=grad_values)

def log_grads_and_vars(grads_and_vars):
    r'''
    Let's also introduce a helper function for logging collections of gradient/variable tuples.
    '''
    for gradient, variable in grads_and_vars:
        log_variable(variable, gradient=gradient)

def export():
    r'''
    Restores the trained variables into a simpler graph that will be exported for serving.
    '''
    log.info('Exporting the model...')
    with tf.device('/cpu:0'):

        tf.reset_default_graph()
        session = tf.Session(config=session_config)

        # Run inference

        # Input tensor will be of shape [batch_size, n_steps, n_input + 2*n_input*n_context]
        input_tensor = tf.placeholder(tf.float32, [None, None, n_input + 2*n_input*n_context], name='input_node')

        seq_length = tf.placeholder(tf.int32, [None], name='input_lengths')

        # Calculate the logits of the batch using BiRNN
        logits = BiRNN(input_tensor, tf.to_int64(seq_length), no_dropout)

        # Beam search decode the batch
        decoded, _ = tf.nn.ctc_beam_search_decoder(logits, seq_length, merge_repeated=False)
        decoded = tf.convert_to_tensor(
            [tf.sparse_tensor_to_dense(sparse_tensor) for sparse_tensor in decoded], name='output_node')

        # TODO: Transform the decoded output to a string

        # Create a saver and exporter using variables from the above newly created graph
        saver = tf.train.Saver(tf.global_variables())
        model_exporter = exporter.Exporter(saver)

        # Restore variables from training checkpoint
        # TODO: This restores the most recent checkpoint, but if we use validation to counterract
        #       over-fitting, we may want to restore an earlier checkpoint.
        checkpoint = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        checkpoint_path = checkpoint.model_checkpoint_path
        saver.restore(session, checkpoint_path)
        log.info('Restored checkpoint at training epoch %d' % (int(checkpoint_path.split('-')[-1]) + 1))

        # Initialise the model exporter and export the model
        model_exporter.init(session.graph.as_graph_def(),
                            named_graph_signatures = {
                                'inputs': exporter.generic_signature(
                                    { 'input': input_tensor,
                                      'input_lengths': seq_length}),
                                'outputs': exporter.generic_signature(
                                    { 'outputs': decoded})})
        if FLAGS.remove_export:
            actual_export_dir = os.path.join(FLAGS.export_dir, '%08d' % FLAGS.export_version)
            if os.path.isdir(actual_export_dir):
                log.info('Removing old export')
                shutil.rmtree(actual_FLAGS.export_dir)
        try:
            # Export serving model
            model_exporter.export(FLAGS.export_dir, tf.constant(FLAGS.export_version), session)

            # Export graph
            input_graph_name = 'input_graph.pb'
            tf.train.write_graph(session.graph, FLAGS.export_dir, input_graph_name, as_text=False)

            # Freeze graph
            input_graph_path = os.path.join(FLAGS.export_dir, input_graph_name)
            input_saver_def_path = ''
            input_binary = True
            output_node_names = 'output_node'
            restore_op_name = 'save/restore_all'
            filename_tensor_name = 'save/Const:0'
            output_graph_path = os.path.join(FLAGS.export_dir, 'output_graph.pb')
            clear_devices = False
            freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                                      input_binary, checkpoint_path, output_node_names,
                                      restore_op_name, filename_tensor_name,
                                      output_graph_path, clear_devices, '')

            log.info('Models exported at %s' % (FLAGS.export_dir))
        except RuntimeError:
            log.error(sys.exc_info()[1])

class ClusterEndPoint(ClusterMessagingClient):
    '''
    Local cluster messaging end point.
    '''
    def __init__(self, cluster_spec, task):
        ClusterMessagingClient.__init__(self, cluster_spec, 'worker', task)
        self._index_lock = Lock()
        self._get_gpus_per_worker_lock = Lock()
        self._index = 0
        self._event = Event()
        # Public member to set an event callback after the model feeder got created.
        # Reason for not being a ctor parameter: Creating the model feeder requires
        # the cluster GPU configuration that in turn already requires messaging to
        # query the GPU configuration of each worker.
        self.on_start_data_set = None
        self.gpus_per_worker = []

    def _start_data_set(self, data_set_index):
        if self.on_start_data_set:
            self.on_start_data_set(data_set_index)

    def start_data_set(self, data_set_index, index_offset=0):
        '''
        Starts a data_set on every node of the cluster.
        Should only be called by chief worker.
        'data_set_index' - 0 for the training set, 1 for the validation set and 2 for the test set
        'index_offset' - sample offset of the first sample within the data set to enqueue
        '''
        self._index = index_offset
        for i in self.cluster_spec.task_indices('worker'):
            self.call('worker', i, '_start_data_set', data_set_index)

    def _allocate_indices(self, number):
        with self._index_lock:
            value = self._index
            self._index = self._index + number
            return value

    def allocate_indices(self, number):
        '''
        Gets an index allocation from chief worker.
        'number' - Number of indices to allocate.
        Returns new base index of allocation.
        '''
        return self.call('worker', 0, '_allocate_indices', number)

    def _get_gpus(self):
        return memory_limits

    def _get_gpus_per_worker(self):
        # lock is required to prevent competing calls to interfer with each other
        with self._get_gpus_per_worker_lock:
            # lazy evaluation
            if len(self.gpus_per_worker) == 0:
                # query each worker for his GPUs
                for i in self.cluster_spec.task_indices('worker'):
                    self.gpus_per_worker.append(self.call('worker', i, '_get_gpus'))
        return self.gpus_per_worker

    def get_gpus_per_worker(self):
        '''
        Retrieves GPU configuration for every worker.
        Returns a list (of workers) of a list of GPUs (of one worker).
        '''
        return self.call('worker', 0, '_get_gpus_per_worker')

    def _quit(self):
        self._event.set()

    def quit(self):
        '''
        Quitting the cluster/training by unblocking every node's wait call.
        Should only be called by chief worker.
        '''
        for i in self.cluster_spec.task_indices('worker'):
            self.call_async('worker', i, '_quit', None)

    def wait(self):
        '''
        Blocking call that waits for the cluster/training to conclude.
        '''
        self._event.wait()

def train() :
    # creating local server
    server = tf.train.Server(cluster_spec, job_name='worker', task_index=FLAGS.task_index)
    # preparing local instances of all required data sets
    train_set = DataSet(FLAGS.train_files,
                        batch_size=FLAGS.train_batch_size,
                        limit=FLAGS.limit_train,
                        skip=FLAGS.skip_train,
                        ascending=FLAGS.train_ascending) if len(FLAGS.train_files) > 0 else None
    dev_set =   DataSet(FLAGS.dev_files,
                        batch_size=FLAGS.dev_batch_size,
                        limit=FLAGS.limit_dev,
                        skip=FLAGS.skip_dev,
                        ascending=FLAGS.dev_ascending) if len(FLAGS.dev_files) > 0 else None
    test_set =  DataSet(FLAGS.test_files,
                        batch_size=FLAGS.test_batch_size,
                        limit=FLAGS.limit_test,
                        skip=FLAGS.skip_test,
                        ascending=FLAGS.test_ascending) if len(FLAGS.test_files) > 0 else None
    data_sets = [train_set, dev_set, test_set]

    with tf.Session(server.target, config=session_config) as session:
        # creating the cluster messaging end point
        cluster = ClusterEndPoint(cluster_spec, FLAGS.task_index)
        # central thread coordinator
        coord = tf.train.Coordinator()
        # starting cluster messaging thrads
        cluster.start_queue_threads(session, coord)
        # query all GPU configurations across the cluster
        gpus_per_worker = cluster.get_gpus_per_worker()
        # finally the local model feeder can be initialized
        model_feeder = ModelFeeder(gpus_per_worker,
                                   FLAGS.task_index,
                                   n_input,
                                   n_context,
                                   alphabet,
                                   FLAGS.threads_per_set,
                                   FLAGS.loader_buffer,
                                   min(memory_limits),
                                   FLAGS.queue_capacity,
                                   allocate_indices=cluster.allocate_indices)
        # react on data set (re-)starting (cluster wide event)
        cluster.on_start_data_set = \
            lambda data_set_index: model_feeder.start_data_set(data_sets[data_set_index])
        # get all relevant graph end-points
        optimizer, results_tuple, gradients, sample_number, mean_edit_distance, loss = \
            get_tower_results(model_feeder)
        # add summaries of all variables and gradients to log
        log_grads_and_vars(gradients)
        # op to merge all summaries for the summary hook
        merge_all_summaries_op = tf.summary.merge_all()
        # apply gradients to modify the model
        apply_gradient_op = optimizer.apply_gradients(gradients)
        # increment global sample counter
        sample_counter = model_variable('sample_counter', None, 0, trainable=False)
        sample_inc_op = tf.assign_add(sample_counter, sample_number)

        # checkpoint manager to deal with all saving, restoring and result logging
        checkpoint_manager = CheckpointManager(checkpoint_dir=FLAGS.checkpoint_dir,
                                               load=FLAGS.load,
                                               inter_secs=FLAGS.inter_secs,
                                               keep_n_inters=FLAGS.keep_n_inters,
                                               keep_n_epochs=FLAGS.keep_n_epochs)

        # start feeding - feeding runs on every node
        # Note: happens to result in very strange errors, if done earlier
        # (supposedly during graph construction)
        model_feeder.start_queue_threads(session, coord)

        # in-graph replication approach
        # all graph evaluation initiated by chief worker only
        if is_chief:
            # let the checkpoint manager either load a model or initialize all variables
            epoch_history = checkpoint_manager.start(session)
            # getting historic validation loss log for early stopping
            dev_losses = [e[2] for e in epoch_history if not e[2] is None]
            # retrieving number of applied samples from (potentially restored) model
            n_samples_trained_on_model = session.run(sample_counter)
            # number of samples per epoch - to be at least 1
            n_samples_per_epoch = len(train_set.files)
            assert n_samples_per_epoch > 0
            # start epoch of our training
            start_epoch = (n_samples_trained_on_model // n_samples_per_epoch) + 1
            # number of samples trained already 'on top of' our start epoch
            n_samples_already_trained = (n_samples_trained_on_model % n_samples_per_epoch)
            # a negative epoch means to add its absolute number to the epochs already computed
            target_epoch = (start_epoch - 1 + abs(FLAGS.epoch)) if FLAGS.epoch < 0 else FLAGS.epoch
            # important debug info
            log.debug('start epoch: %d' % start_epoch)
            log.debug('target epoch: %d' % target_epoch)
            log.debug('number of samples per epoch: %d' % n_samples_per_epoch)
            log.debug('number of samples already trained in start epoch: %d' % n_samples_already_trained)

            def apply_set(data_set_index, should_train, should_report, label, offset=0):
                # start feeding requested data set on every worker
                cluster.start_data_set(data_set_index, offset)
                # training operation in case of training requested
                train_op = [apply_gradient_op] if should_train else []
                # global sample counter incrementation and/or retrieval
                overall_samples_op = sample_inc_op if should_train else sample_counter
                # requirements for computing a WER report
                if should_report:
                    # initialize mean edit distance aggregator
                    total_mean_edit_distance = 0.0
                    # create report results tuple
                    report_results = ([],[],[],[])
                    # extend the session.run parameters
                    report_params = [results_tuple, mean_edit_distance]
                else:
                    report_params = []
                # initializing all aggregators
                n_samples_applied = 0
                total_loss = 0
                total_mean_edit_distance = 0
                # total number of samples to apply for current data set
                n_samples_to_apply = len(data_sets[data_set_index].files)
                # looping over batches till every sample got applied
                while offset + n_samples_applied < n_samples_to_apply:
                    # run one step
                    _, n_samples_trained_on_model, n_samples_in_step, current_loss, current_report = \
                        session.run([train_op, overall_samples_op, sample_number, loss, report_params])
                    # collect results
                    n_samples_applied += n_samples_in_step
                    log.step('Applied %d samples (%d of %d).' % \
                        (n_samples_in_step, offset + n_samples_applied, n_samples_to_apply))
                    # aggregate loss (weighted by number of samples)
                    total_loss += current_loss * n_samples_in_step
                    if should_report:
                        samples, current_mean_edit_distance = current_report
                        # collect individual sample results
                        for i in range(len(model_feeder.towers)):
                            # collect the labels
                            report_results[0].extend(sparse_tensor_value_to_texts(samples[0][i], alphabet))
                            # collect the decodings - at the moment we default to the first one
                            report_results[1].extend(sparse_tensor_value_to_texts(samples[1][i][0], alphabet))
                            # collect the distances
                            report_results[2].extend(samples[2][i])
                            # collect the losses
                            report_results[3].extend(samples[3][i])
                        # aggregate mean edit distance (weighted by number of samples)
                        total_mean_edit_distance += current_mean_edit_distance * n_samples_in_step
                    if should_train:
                        # give checkpoint manager a chance to write an intermediate checkpoint
                        checkpoint_manager.step(session, n_samples_trained_on_model)
                # gathering results (dividing weighted aggregates by total number of samples)
                total_loss = total_loss / n_samples_applied
                if should_report:
                    total_mean_edit_distance = total_mean_edit_distance / n_samples_applied
                    samples = []
                    mean_wer = 0.0
                    # re-arrange list and exclude dummy samples
                    report_results = list(zip(*report_results))
                    print('LEN report_results: %r' % report_results)
                    report_results = [r for r in report_results if r[0] != ' ']
                    # do spell-checking and compute WER - only keep samples with WER > 0
                    for s_label, s_decoding, s_distance, s_loss in report_results:
                        s_wer = wer(s_label, s_decoding)
                        mean_wer += s_wer
                        if s_wer > 0:
                            samples.append((s_label, s_decoding, s_loss, s_distance, s_wer))
                    # get the mean WER from the accumulated one
                    mean_wer = mean_wer / len(report_results)
                    # order the remaining samples by their loss (lowest loss on top)
                    samples.sort(key=lambda s: s[2])
                    # take only the first report_count samples
                    samples = samples[:FLAGS.report_count]
                    # order the remaining samples by their WER (lowest WER on top)
                    samples.sort(key=lambda s: s[4])
                    # print the report
                    splitter = '  ' + '-' * 20
                    log.info('%s result - average WER: %.2f, average loss: %.2f, average edit distance: %.2f' % \
                             (label, mean_wer, total_loss, s_distance))
                    log.info(splitter)
                    for s_label, s_decoding, s_loss, s_distance, s_wer in samples:
                        log.info('- WER: %.2f, loss: %.2f, edit distance: %.2f' % (s_wer, s_loss, s_distance))
                        log.info('  expected: %s' % s_label)
                        log.info('  decoded:  %s' % s_decoding)
                        log.info(splitter)
                else:
                    log.info('%s result - loss: %.2f' % (label, total_loss))
                return total_loss, n_samples_trained_on_model

            if FLAGS.train and target_epoch >= start_epoch:
                log.info('STARTING Optimization')
                for epoch in range(start_epoch, target_epoch + 1):
                    # if early stopping is enabled and possible...
                    if FLAGS.early_stop and FLAGS.validation_step > 0 and len(dev_losses) >= FLAGS.es_nsteps:
                        # calculate the mean of losses for past epochs
                        mean_loss = np.mean(dev_losses[-FLAGS.es_nsteps:-1])
                        # calculate the standard deviation for losses from validation part in the past epochs
                        std_loss = np.std(dev_losses[-FLAGS.es_nsteps:-1])
                        # update the list of losses incurred
                        dev_losses = dev_losses[-FLAGS.es_nsteps:]
                        es_info = 'steps: %d, validation loss: %f, standard deviation: %f, mean: %f' % \
                                  (FLAGS.es_nsteps, dev_losses[-1], std_loss, mean_loss)
                        log.debug('Checking for early stopping - %s' % es_info)
                        # check if validation loss has started increasing or is not decreasing substantially,
                        # making sure slight fluctuations don't bother the early stopping from working
                        if (dev_losses[-1] > np.max(dev_losses[:-1])) or \
                           (abs(dev_losses[-1] - mean_loss) < FLAGS.es_mean_thresh and std_loss < FLAGS.es_std_thresh):
                            log.info('Early stop triggered - %s' % es_info)
                            # early stopping -> just exit loop
                            break
                    log.info('=' * 100)
                    # training
                    log.info('Training epoch %d...' % epoch)
                    should_report = FLAGS.display_step > 0 and (epoch % FLAGS.display_step) == 0
                    train_loss, n_samples_trained_on_model = apply_set(0, True, should_report, 'Training', offset=n_samples_already_trained)
                    n_samples_already_trained = 0
                    # validation
                    dev_loss = None
                    if FLAGS.validation_step > 0 and epoch % FLAGS.validation_step == 0:
                        log.info('Validating epoch %d...' % epoch)
                        dev_loss, _ = apply_set(1, False, should_report, 'Validation')
                        dev_losses.append(dev_loss)
                    # let the checkpoint manager log results and do an epoch checkpoint of the current model
                    checkpoint_manager.epoch(session, epoch, n_samples_trained_on_model, train_loss, dev_loss=dev_loss)
                    log.info('Finished epoch %d.' % epoch)
                log.info('=' * 100)
                log.info('FINISHED Optimization')

            if FLAGS.test:
                # test
                log.info('Testing epoch %d...' % target_epoch)
                apply_set(2, False, True, 'Test')
                log.info('Finished testing epoch %d.' % target_epoch)

            log.debug('Quitting cluster...')
            cluster.quit()
        else:
            cluster.wait()
        log.debug('Shutting down training session...')
        # requesting threads to stop
        coord.request_stop()
        # closing feeder queues
        model_feeder.close_queues(session)
        # closing cluster end point queues
        cluster.close_queues(session)
        # waiting for coordinated threads to stop
        coord.join()
    log.debug('Training session closed.')

def main(_) :
    # initializing all global variables (mostly from FLAGS values)
    initialize_globals()
    if FLAGS.train or FLAGS.test:
        train()
    if FLAGS.export_dir and is_chief:
        export()

if __name__ == '__main__' :
    tf.app.run()
