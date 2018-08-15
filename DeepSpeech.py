#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import os
import sys

log_level_index = sys.argv.index('--log_level') + 1 if '--log_level' in sys.argv else 0
os.environ['TF_CPP_MIN_LOG_LEVEL'] = sys.argv[log_level_index] if log_level_index > 0 and log_level_index < len(sys.argv) else '3'

import datetime
import pickle
import shutil
import six
import subprocess
import tensorflow as tf
import time
import datetime
import traceback
import inspect
import math
import glob

from six.moves import zip, range, filter, urllib, BaseHTTPServer
from tensorflow.python.tools import freeze_graph
from threading import Thread, Lock
from util.audio import audiofile_to_input_vector
from util.feeding import DataSet, ModelFeeder
from util.gpu import get_available_gpus
from util.shared_lib import check_cupti
from util.text import sparse_tensor_value_to_texts, wer, levenshtein, Alphabet, ndarray_to_text
from xdg import BaseDirectory as xdg
import numpy as np
import horovod.tensorflow as hvd

from tensorflow.python.framework import ops

from tensorflow.python.client import timeline

#
# Importer
# ========

tf.app.flags.DEFINE_string  ('train_files',      '',          'comma separated list of files specifying the dataset used for training. multiple files will get merged')
tf.app.flags.DEFINE_string  ('dev_files',        '',          'comma separated list of files specifying the dataset used for validation. multiple files will get merged')
tf.app.flags.DEFINE_string  ('test_files',       '',          'comma separated list of files specifying the dataset used for testing. multiple files will get merged')
tf.app.flags.DEFINE_boolean ('fulltrace',        False,       'if full trace debug info should be generated during training')
tf.app.flags.DEFINE_string  ('trace_filename',       '',          'Name of the json trace file to log the TF Timeline results')

tf.app.flags.DEFINE_integer ('iters_per_worker', 1,           'number of train or inference iterations per worker before results are sent back')

tf.app.flags.DEFINE_integer ('inter_op_threads', 2,           'number of inter op parallelism threads')
tf.app.flags.DEFINE_integer ('intra_op_threads', 4,           'number of intra op parallelism threads')

# Global Constants
# ================

tf.app.flags.DEFINE_boolean ('train',            True,        'whether to train the network')
tf.app.flags.DEFINE_boolean ('test',             False,        'whether to test the network')
tf.app.flags.DEFINE_integer ('epoch',            15,          'target epoch to train - if negative, the absolute number of additional epochs will be trained')

tf.app.flags.DEFINE_boolean ('lrn_warmup',            False,          'Allow gradual warmup of the learning rate for large batch size training')

tf.app.flags.DEFINE_integer ('warmup_epochs',            5,          'Number of epochs of gradual warmup for large batch size training')

tf.app.flags.DEFINE_boolean ('use_warpctc',      False,       'whether to use GPU bound Warp-CTC')

tf.app.flags.DEFINE_float   ('dropout_rate',     0.30,        'dropout rate for feedforward layers')
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
tf.app.flags.DEFINE_float   ('learning_rate',    0.0016,       'learning rate of Adam optimizer')

# Batch sizes

tf.app.flags.DEFINE_integer ('train_batch_size', 12,           'number of elements in a training batch')
tf.app.flags.DEFINE_integer ('dev_batch_size',   1,           'number of elements in a validation batch')
tf.app.flags.DEFINE_integer ('test_batch_size',  1,           'number of elements in a test batch')

# Checkpointing

tf.app.flags.DEFINE_string  ('checkpoint_dir',   '',          'directory in which checkpoints are stored - defaults to directory "deepspeech/checkpoints" within user\'s data home specified by the XDG Base Directory Specification')
tf.app.flags.DEFINE_integer ('checkpoint_secs',  600,         'checkpoint saving interval in seconds')
tf.app.flags.DEFINE_integer ('max_to_keep',      5,           'number of checkpoint files to keep - default value is 5')

# Exporting

tf.app.flags.DEFINE_string  ('export_dir',       '',          'directory in which exported models are stored - if omitted, the model won\'t get exported')
tf.app.flags.DEFINE_integer ('export_version',   1,           'version number of the exported model')
tf.app.flags.DEFINE_boolean ('remove_export',    False,       'whether to remove old exported models')
tf.app.flags.DEFINE_boolean ('use_seq_length',   True,        'have sequence_length in the exported graph (will make tfcompile unhappy)')

# Reporting
tf.app.flags.DEFINE_integer ('display_step',     0,           'number of epochs we cycle through before displaying detailed progress - 0 means no progress display')
tf.app.flags.DEFINE_integer ('log_level',        1,           'log level for console logs - 0: INFO, 1: WARN, 2: ERROR, 3: FATAL')
tf.app.flags.DEFINE_boolean ('log_traffic',      False,       'log cluster transaction and traffic information during debug logging')

tf.app.flags.DEFINE_boolean ('log_placement',    False,       'whether to log device placement of the operators to the console')
tf.app.flags.DEFINE_integer ('report_count',     10,          'number of phrases with lowest WER (best matching) to print out during a WER report')

tf.app.flags.DEFINE_string  ('summary_dir',      '',          'target directory for TensorBoard summaries - defaults to directory "deepspeech/summaries" within user\'s data home specified by the XDG Base Directory Specification')
tf.app.flags.DEFINE_integer ('summary_secs',     0,           'interval in seconds for saving TensorBoard summaries - if 0, no summaries will be written')

# Geometry

tf.app.flags.DEFINE_integer ('n_hidden',         2048,        'layer width to use when initialising layers')

# Initialization

tf.app.flags.DEFINE_integer ('random_seed',      4567,        'default random seed that is used to initialize variables')
tf.app.flags.DEFINE_float   ('default_stddev',   0.046875,    'default standard deviation to use when initialising weights and biases')

# Decoder

tf.app.flags.DEFINE_string  ('decoder_library_path', 'native_client/libctc_decoder_with_kenlm.so', 'path to the libctc_decoder_with_kenlm.so library containing the decoder implementation.')
tf.app.flags.DEFINE_string  ('alphabet_config_path', 'data/alphabet.txt', 'path to the configuration file specifying the alphabet used by the network. See the comment in data/alphabet.txt for a description of the format.')
tf.app.flags.DEFINE_string  ('lm_binary_path',       'data/lm/lm.binary', 'path to the language model binary file created with KenLM')
tf.app.flags.DEFINE_string  ('lm_trie_path',         'data/lm/trie', 'path to the language model trie file created with native_client/generate_trie')
tf.app.flags.DEFINE_integer ('beam_width',        1024,       'beam width used in the CTC decoder when building candidate transcriptions')
tf.app.flags.DEFINE_float   ('lm_weight',         1.75,       'the alpha hyperparameter of the CTC decoder. Language Model weight.')
tf.app.flags.DEFINE_float   ('word_count_weight', 1.00,      'the beta hyperparameter of the CTC decoder. Word insertion weight (penalty).')
tf.app.flags.DEFINE_float   ('valid_word_count_weight', 1.00, 'valid word insertion weight. This is used to lessen the word insertion penalty when the inserted word is part of the vocabulary.')

# Inference mode

tf.app.flags.DEFINE_string  ('one_shot_infer',       '',       'one-shot inference mode: specify a wav file and the script will load the checkpoint and perform inference on it. Disables training, testing and exporting.')

# Initialize from frozen model

tf.app.flags.DEFINE_string  ('initialize_from_frozen_model', '', 'path to frozen model to initialize from. This behaves like a checkpoint, loading the weights from the frozen model and starting training with those weights. The optimizer parameters aren\'t restored, so remember to adjust the learning rate.')

# For averaging gradients
tf.app.flags.DEFINE_string  ('job_name',         'localhost', 'job name - one of localhost (default), worker, ps')
tf.app.flags.DEFINE_integer ('task_index',       0,           'index of task within the job - worker with index 0 will be the chief')

for var in ['b1', 'h1', 'b2', 'h2', 'b3', 'h3', 'b5', 'h5', 'b6', 'h6']:
    tf.app.flags.DEFINE_float('%s_stddev' % var, None, 'standard deviation to use when initialising %s' % var)

FLAGS = tf.app.flags.FLAGS

def initialize_globals():

    hvd.init()

    # Determine, if we are the chief worker
    global is_chief
    is_chief = hvd.rank()==0

    # The absolute number of computing nodes - regardless of cluster or single mode
    global num_workers
    num_workers = max(1, hvd.size())

    # The rank of the worker and size of the cluster
    global rank
    global size
    if num_workers == 1:
        rank = 0
        size = 1
    else:
        rank = hvd.rank() 
        size = hvd.size()

    #The epoch object to be used for training
    global epoch
    epoch = None

    #The epoch id to be used for training
    global epoch_id
    epoch_id = 0

    #The total number of training jobs
    global num_jobs_train
    num_jobs_train = 0

    # The device path base for this node
    global worker_device
    # if num_workers == 1:
    #     worker_device = '/job:%s/task:%d' % ('localhost', 0)
    # else:
    worker_device = '/job:%s/task:%d' % ('worker', hvd.rank())
    
    # This node's CPU device
    global cpu_device
    #cpu_device = worker_device + '/cpu:0'
    cpu_device = '/job:%s/task:%d/cpu:0' % (FLAGS.job_name, FLAGS.task_index)

    # This node's available GPU devices
    global available_devices
    available_devices = [worker_device + gpu for gpu in get_available_gpus()]

    # If there is no GPU available, we fall back to CPU based operation
    if 0 == len(available_devices):
        available_devices = [cpu_device]

    # Set default dropout rates
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

    # Set default checkpoint dir
    if len(FLAGS.checkpoint_dir) == 0:
        FLAGS.checkpoint_dir = xdg.save_data_path(os.path.join('deepspeech','checkpoints'))

    # Set default summary dir
    if len(FLAGS.summary_dir) == 0:
        FLAGS.summary_dir = xdg.save_data_path(os.path.join('deepspeech','summaries'))

    # Standard session configuration that'll be used for all new sessions.
    global session_config
    session_config = tf.ConfigProto(inter_op_parallelism_threads=FLAGS.inter_op_threads,intra_op_parallelism_threads=FLAGS.intra_op_threads)

    global alphabet
    alphabet = Alphabet(os.path.abspath(FLAGS.alphabet_config_path))

    # Geometric Constants
    # ===================

    # For an explanation of the meaning of the geometric constants, please refer to
    # doc/Geometry.md

    # Number of MFCC features
    global n_input
    n_input = 26 # TODO: Determine this programatically from the sample rate

    # The number of frames in the context
    global n_context
    n_context = 9 # TODO: Determine the optimal value using a validation data set

    # Number of units in hidden layers
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

    # The number of units in the third layer, which feeds in to the LSTM
    global n_hidden_3
    n_hidden_3 = 2 * n_cell_dim

    # The number of characters in the target language plus one
    global n_character
    n_character = alphabet.size() + 1 # +1 for CTC blank label

    # The number of units in the sixth layer
    global n_hidden_6
    n_hidden_6 = n_character

    # Assign default values for standard deviation
    for var in ['b1', 'h1', 'b2', 'h2', 'b3', 'h3', 'b5', 'h5', 'b6', 'h6']:
        val = getattr(FLAGS, '%s_stddev' % var)
        if val is None:
            setattr(FLAGS, '%s_stddev' % var, FLAGS.default_stddev)

    if len(FLAGS.one_shot_infer) > 0:
        FLAGS.train = False
        FLAGS.test = False
        FLAGS.export_dir = ''
        if not os.path.exists(FLAGS.one_shot_infer):
            log_error('Path specified in --one_shot_infer is not a valid file.')
            exit(1)

    if not os.path.exists(os.path.abspath(FLAGS.decoder_library_path)):
        print('ERROR: The decoder library file does not exist. Make sure you have ' \
              'downloaded or built the native client binaries and pass the ' \
              'appropriate path to the binaries in the --decoder_library_path parameter.')

    global custom_op_module
    custom_op_module = tf.load_op_library(FLAGS.decoder_library_path)

# Logging functions
# =================

def prefix_print(prefix, message):
    print(prefix + ('\n' + prefix).join(message.split('\n')))

def log_debug(message):
    if FLAGS.log_level == 0:
        prefix_print('D ', message)

def log_traffic(message):
    if FLAGS.log_traffic:
        log_debug(message)

def log_info(message):
    if FLAGS.log_level <= 1:
        prefix_print('I ', message)

def log_warn(message):
    if FLAGS.log_level <= 2:
        prefix_print('W ', message)

def log_error(message):
    if FLAGS.log_level <= 3:
        prefix_print('E ', message)


# Graph Creation
# ==============

def variable_on_worker_level(name, shape, initializer):
    r'''
    Next we concern ourselves with graph creation.
    However, before we do so we must introduce a utility function ``variable_on_worker_level()``
    used to create a variable in CPU memory.
    '''
    with tf.device('/cpu:0'):
        # Create or get apropos variable
        var = tf.get_variable(name=name, shape=shape, initializer=initializer)
    return var

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
    b1 = variable_on_worker_level('b1', [n_hidden_1], tf.random_normal_initializer(stddev=FLAGS.b1_stddev))
    h1 = variable_on_worker_level('h1', [n_input + 2*n_input*n_context, n_hidden_1], tf.contrib.layers.xavier_initializer(uniform=False))
    layer_1 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(batch_x, h1), b1)), FLAGS.relu_clip)
    layer_1 = tf.nn.dropout(layer_1, (1.0 - dropout[0]))

    # 2nd layer
    b2 = variable_on_worker_level('b2', [n_hidden_2], tf.random_normal_initializer(stddev=FLAGS.b2_stddev))
    h2 = variable_on_worker_level('h2', [n_hidden_1, n_hidden_2], tf.random_normal_initializer(stddev=FLAGS.h2_stddev))
    layer_2 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(layer_1, h2), b2)), FLAGS.relu_clip)
    layer_2 = tf.nn.dropout(layer_2, (1.0 - dropout[1]))

    # 3rd layer
    b3 = variable_on_worker_level('b3', [n_hidden_3], tf.random_normal_initializer(stddev=FLAGS.b3_stddev))
    h3 = variable_on_worker_level('h3', [n_hidden_2, n_hidden_3], tf.random_normal_initializer(stddev=FLAGS.h3_stddev))
    layer_3 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(layer_2, h3), b3)), FLAGS.relu_clip)
    layer_3 = tf.nn.dropout(layer_3, (1.0 - dropout[2]))

    # Now we create the forward and backward LSTM units.
    # Both of which have inputs of length `n_cell_dim` and bias `1.0` for the forget gate of the LSTM.

    # Forward direction cell:
    lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(n_cell_dim, forget_bias=1.0, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)

    lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(lstm_fw_cell,
                                                input_keep_prob=1.0 - dropout[3],
                                                output_keep_prob=1.0 - dropout[3],
                                                seed=FLAGS.random_seed)
    # Backward direction cell:
    lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(n_cell_dim, forget_bias=1.0, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)
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
    b5 = variable_on_worker_level('b5', [n_hidden_5], tf.random_normal_initializer(stddev=FLAGS.b5_stddev))
    h5 = variable_on_worker_level('h5', [(2 * n_cell_dim), n_hidden_5], tf.random_normal_initializer(stddev=FLAGS.h5_stddev))
    layer_5 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(outputs, h5), b5)), FLAGS.relu_clip)
    layer_5 = tf.nn.dropout(layer_5, (1.0 - dropout[5]))

    # Now we apply the weight matrix `h6` and bias `b6` to the output of `layer_5`
    # creating `n_classes` dimensional vectors, the logits.
    b6 = variable_on_worker_level('b6', [n_hidden_6], tf.random_normal_initializer(stddev=FLAGS.b6_stddev))
    h6 = variable_on_worker_level('h6', [n_hidden_5, n_hidden_6], tf.contrib.layers.xavier_initializer(uniform=False))
    layer_6 = tf.add(tf.matmul(layer_5, h6), b6)

    # Finally we reshape layer_6 from a tensor of shape [n_steps*batch_size, n_hidden_6]
    # to the slightly more useful shape [n_steps, batch_size, n_hidden_6].
    # Note, that this differs from the input in that it is time-major.
    layer_6 = tf.reshape(layer_6, [-1, batch_x_shape[0], n_hidden_6], name="logits")

    # Output shape: [n_steps, batch_size, n_hidden_6]
    return layer_6

def decode_with_lm(inputs, sequence_length, beam_width=100,
                   top_paths=1, merge_repeated=True):
  decoded_ixs, decoded_vals, decoded_shapes, log_probabilities = (
      custom_op_module.ctc_beam_search_decoder_with_lm(
          inputs, sequence_length, beam_width=beam_width,
          model_path=FLAGS.lm_binary_path, trie_path=FLAGS.lm_trie_path, alphabet_path=FLAGS.alphabet_config_path,
          lm_weight=FLAGS.lm_weight, valid_word_count_weight=FLAGS.valid_word_count_weight,
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

def calculate_mean_edit_distance_and_loss(model_feeder, tower, dropout):
    r'''
    This routine beam search decodes a mini-batch and calculates the loss and mean edit distance.
    Next to total and average loss it returns the mean edit distance,
    the decoded result and the batch's original Y.
    '''
    # Obtain the next batch of data
    batch_x, batch_seq_len, batch_y = model_feeder.next_batch(0)

    # Calculate the logits of the batch using BiRNN
    logits = BiRNN(batch_x, tf.to_int64(batch_seq_len), dropout)

    # Compute the CTC loss using either TensorFlow's `ctc_loss` or Baidu's `warp_ctc_loss`.
    if FLAGS.use_warpctc:
        total_loss = tf.contrib.warpctc.warp_ctc_loss(labels=batch_y, inputs=logits, sequence_length=batch_seq_len)
    else:
        total_loss = tf.nn.ctc_loss(labels=batch_y, inputs=logits, sequence_length=batch_seq_len)

    # Calculate the average loss across the batch
    avg_loss = tf.reduce_mean(total_loss)

    # Beam search decode the batch
    decoded, _ = decode_with_lm(logits, batch_seq_len, merge_repeated=False, beam_width=FLAGS.beam_width)

    # Compute the edit (Levenshtein) distance
    distance = tf.edit_distance(tf.cast(decoded[0], tf.int32), batch_y)

    # Compute the mean edit distance
    mean_edit_distance = tf.reduce_mean(distance)

    # Finally we return the
    # - calculated total and
    # - average losses,
    # - the Levenshtein distance,
    # - the recognition mean edit distance,
    # - the decoded batch and
    # - the original batch_y (which contains the verified transcriptions).
    return total_loss, avg_loss, distance, mean_edit_distance, decoded, batch_y


# Adam Optimization
# =================

# In contrast to 'Deep Speech: Scaling up end-to-end speech recognition'
# (http://arxiv.org/abs/1412.5567),
# in which 'Nesterov's Accelerated Gradient Descent'
# (www.cs.toronto.edu/~fritz/absps/momentum.pdf) was used,
# we will use the Adam method for optimization (http://arxiv.org/abs/1412.6980),
# because, generally, it requires less fine-tuning.
def create_optimizer(lrn_rate):
    tf.summary.scalar('learning_rate', lrn_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate=lrn_rate,
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
def get_tower_results(model_feeder, optimizer):
    r'''
    With this preliminary step out of the way, we can for each GPU introduce a
    tower for which's batch we calculate

    * The CTC decodings ``decoded``,
    * The (total) loss against the outcome (Y) ``total_loss``,
    * The loss averaged over the whole batch ``avg_loss``,
    * The optimization gradient (computed based on the averaged loss),
    * The Levenshtein distances between the decodings and their transcriptions ``distance``,
    * The mean edit distance of the outcome averaged over the whole batch ``mean_edit_distance``

    and retain the original ``labels`` (Y).
    ``decoded``, ``labels``, the optimization gradient, ``distance``, ``mean_edit_distance``,
    ``total_loss`` and ``avg_loss`` are collected into the corresponding arrays
    ``tower_decodings``, ``tower_labels``, ``tower_gradients``, ``tower_distances``,
    ``tower_mean_edit_distances``, ``tower_total_losses``, ``tower_avg_losses`` (dimension 0 being the tower).
    Finally this new method ``get_tower_results()`` will return those tower arrays.
    In case of ``tower_mean_edit_distances`` and ``tower_avg_losses``, it will return the
    averaged values instead of the arrays.
    '''
    # Tower labels to return
    tower_labels = []

    # Tower decodings to return
    tower_decodings = []

    # Tower distances to return
    tower_distances = []

    # Tower total batch losses to return
    tower_total_losses = []

    # Tower gradients to return
    tower_gradients = []

    # To calculate the mean of the mean edit distances
    tower_mean_edit_distances = []

    # To calculate the mean of the losses
    tower_avg_losses = []

    with tf.variable_scope(tf.get_variable_scope()):
        # Loop over available_devices
        for i in range(len(available_devices)):
            # Execute operations of tower i on device i
            if len(available_devices) == 0:
                device = tf.train.replica_device_setter(worker_device=available_devices[i])
            else:
                device = tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % hvd.rank())
            with tf.device(device):
                # Create a scope for all operations of tower i
                with tf.name_scope('tower_%d' % i) as scope:
                    # Calculate the avg_loss and mean_edit_distance and retrieve the decoded
                    # batch along with the original batch's labels (Y) of this tower
                    total_loss, avg_loss, distance, mean_edit_distance, decoded, labels = \
                        calculate_mean_edit_distance_and_loss(model_feeder, i, no_dropout if optimizer is None else dropout_rates)

                    # Allow for variables to be re-used by the next tower
                    tf.get_variable_scope().reuse_variables()

                    # Retain tower's labels (Y)
                    tower_labels.append(labels)

                    # Retain tower's decoded batch
                    tower_decodings.append(decoded)

                    # Retain tower's distances
                    tower_distances.append(distance)

                    # Retain tower's total losses
                    tower_total_losses.append(total_loss)

                    # Compute gradients for model parameters using tower's mini-batch
                    gradients = optimizer.compute_gradients(avg_loss)

                    # Retain tower's gradients
                    tower_gradients.append(gradients)

                    # Retain tower's mean edit distance
                    tower_mean_edit_distances.append(mean_edit_distance)

                    # Retain tower's avg losses
                    tower_avg_losses.append(avg_loss)

    # Return the results tuple, the gradients, and the means of mean edit distances and losses
    return (tower_labels, tower_decodings, tower_distances, tower_total_losses), \
           tower_gradients, \
           tf.reduce_mean(tower_mean_edit_distances, 0), \
           tf.reduce_mean(tower_avg_losses, 0)

def average_gradients(tower_gradients):
    r'''
    A routine for computing each variable's average of the gradients obtained from the GPUs.
    Note also that this code acts as a synchronization point as it requires all
    GPUs to be finished with their mini-batch before it can run to completion.
    '''
    # List of average gradients to return to the caller
    average_grads = []

    # Run this on cpu_device to conserve GPU memory
    with tf.device(cpu_device):
        # Loop over gradient/variable pairs from all towers
        for grad_and_vars in zip(*tower_gradients):
            # Introduce grads to store the gradients for the current variable
            grads = []

            # Loop over the gradients for the current variable
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)
                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

            # Average over the 'tower' dimension
            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(grad, 0)

            # Create a gradient/variable tuple for the current variable with its average gradient
            grad_and_var = (grad, grad_and_vars[0][1])

            # Add the current tuple to average_grads
            average_grads.append(grad_and_var)

    # Return result to caller
    return average_grads

# Logging
# =======

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

def get_git_revision_hash():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip()

def get_git_branch():
    return subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).strip()


# For reporting we also need a standard way to do time measurements.
def stopwatch(start_duration=0):
    r'''
    This function will toggle a stopwatch.
    The first call starts it, second call stops it, third call continues it etc.
    So if you want to measure the accumulated time spent in a certain area of the code,
    you can surround that code by stopwatch-calls like this:

    .. code:: python

        fun_time = 0 # initializes a stopwatch
        [...]
        for i in range(10):
          [...]
          # Starts/continues the stopwatch - fun_time is now a point in time (again)
          fun_time = stopwatch(fun_time)
          fun()
          # Pauses the stopwatch - fun_time is now a duration
          fun_time = stopwatch(fun_time)
        [...]
        # The following line only makes sense after an even call of :code:`fun_time = stopwatch(fun_time)`.
        print 'Time spent in fun():', format_duration(fun_time)

    '''
    if start_duration == 0:
        return datetime.datetime.utcnow()
    else:
        return datetime.datetime.utcnow() - start_duration

def format_duration(duration):
    '''Formats the result of an even stopwatch call as hours:minutes:seconds'''
    duration = duration if isinstance(duration, int) else duration.seconds
    m, s = divmod(duration, 60)
    h, m = divmod(m, 60)
    return '%d:%02d:%02d' % (h, m, s)

# Execution
# =========

# String constants for different services of the web handler
PREFIX_NEXT_INDEX = '/next_index_'
PREFIX_GET_JOB = '/get_job_'

# Global ID counter for all objects requiring an ID
id_counter = 0

def new_id():
    '''Returns a new ID that is unique on process level. Not thread-safe.

    Returns:
        int. The new ID
    '''
    global id_counter
    id_counter += 1
    return id_counter

class Sample(object):
    def __init__(self, src, res, loss, mean_edit_distance, sample_wer):
        '''Represents one item of a WER report.

        Args:
            src (str): source text
            res (str): resulting text
            loss (float): computed loss of this item
            mean_edit_distance (float): computed mean edit distance of this item
        '''
        self.src = src
        self.res = res
        self.loss = loss
        self.mean_edit_distance = mean_edit_distance
        self.wer = sample_wer

    def __str__(self):
        return 'WER: %f, loss: %f, mean edit distance: %f\n - src: "%s"\n - res: "%s"' % (self.wer, self.loss, self.mean_edit_distance, self.src, self.res)

class WorkerJob(object):
    def __init__(self, epoch_id, index, set_name, steps, report):
        '''Represents a job that should be executed by a worker.

        Args:
            epoch_id (int): the ID of the 'parent' epoch
            index (int): the epoch index of the 'parent' epoch
            set_name (str): the name of the data-set - one of 'train', 'dev', 'test'
            steps (int): the number of `session.run` calls
            report (bool): if this job should produce a WER report
        '''
        self.id = new_id() 
        self.epoch_id = epoch_id
        self.index = index
        self.worker = -1
        self.set_name = set_name
        self.steps = steps
        self.report = report
        self.loss = -1
        self.mean_edit_distance = -1
        self.wer = -1
        self.samples = []

    def __str__(self):
        return 'Job (ID: %d, worker: %d, epoch: %d, set_name: %s)' % (self.id, self.worker, self.index, self.set_name)

class Epoch(object):
    '''Represents an epoch that should be executed by the Training Coordinator.
    Creates `num_jobs` `WorkerJob` instances in state 'open'.

    Args:
        index (int): the epoch index of the 'parent' epoch
        num_jobs (int): the number of jobs in this epoch

    Kwargs:
        set_name (str): the name of the data-set - one of 'train', 'dev', 'test'
        report (bool): if this job should produce a WER report
    '''
    def __init__(self, index, num_jobs, set_name='train', report=False):
        self.id = new_id()
        self.index = index
        self.num_jobs = num_jobs
        self.set_name = set_name
        self.report = report
        self.wer = -1
        self.loss = -1
        self.mean_edit_distance = -1
        self.jobs_open = []
        self.jobs_running = []
        self.jobs_done = []
        self.samples = []
        for i in range(self.num_jobs):
            self.jobs_open.append(WorkerJob(self.id, self.index, self.set_name, FLAGS.iters_per_worker, self.report))

    def name(self):
        '''Gets a printable name for this epoch.

        Returns:
            str. printable name for this epoch
        '''
        if self.index >= 0:
            ename = ' of Epoch %d' % self.index
        else:
            ename = ''
        if self.set_name == 'train':
            return 'Training%s' % ename
        elif self.set_name == 'dev':
            return 'Validation%s' % ename
        else:
            return 'Test%s' % ename

    def get_job(self, worker):
        '''Gets the next open job from this epoch. The job will be marked as 'running'.

        Args:
            worker (int): index of the worker that takes the job

        Returns:
            WorkerJob. job that has been marked as running for this worker
        '''
        if len(self.jobs_open) > 0:
            job = self.jobs_open.pop(0)
            self.jobs_running.append(job)
            job.worker = worker
            return job
        else:
            return None

def gradual_warmup_then_dec(learning_rate, warmup_steps, end_learning_rate, global_step, steps, name=None):
  with ops.name_scope(name, "gradual_warmup_then_dec",
                      [learning_rate, warmup_steps, end_learning_rate, global_step, steps, name]) as name:

    def warmup_decay(learning_rate, global_step, warmup_steps, end_learning_rate):
      p = global_step / warmup_steps
      diff = tf.subtract(end_learning_rate, learning_rate)
      res = tf.add(learning_rate, tf.multiply(diff, p))
      return res

    if global_step is None:
      raise ValueError("global_step is required for warmup_decay.")

    learning_rate = ops.convert_to_tensor(learning_rate, name="learning_rate")
    end_learning_rate = ops.convert_to_tensor(end_learning_rate, name="end_learning_rate")
    dtype = learning_rate.dtype
    global_step = tf.cast(global_step, dtype)
    warmup_steps = tf.cast(warmup_steps, dtype)
    
    lr = tf.cond(tf.less(global_step, warmup_steps),
                        lambda: warmup_decay(learning_rate, global_step, warmup_steps, end_learning_rate),
                        lambda: tf.train.polynomial_decay(end_learning_rate, global_step-warmup_steps, steps-warmup_steps, 0, power=1))
    return lr

def train(server=None, cluster=None):
    r'''
    Trains the network on a set of nodes on a cluster.
    If no server provided, it performs single process training.
    '''

    # Create a variable to hold the global_step.
    # It will automagically get incremented by the optimizer.
    global_step = tf.Variable(0, trainable=False, name='global_step')

    #Reading the train set
    train_set = DataSet(FLAGS.train_files.split(','),
                        FLAGS.train_batch_size)

    # Reading validation set
    dev_set = DataSet(FLAGS.dev_files.split(','),
                      FLAGS.dev_batch_size)

    # Reading test set
    test_set = DataSet(FLAGS.test_files.split(','),
                       FLAGS.test_batch_size)

    # Combining all sets to a multi set model feeder
    model_feeder = ModelFeeder(train_set,
                               dev_set,
                               test_set,
                               n_input,
                               n_context,
                               alphabet,
                               is_train=FLAGS.train,
                               tower_feeder_count=-1,
                               rank=rank,
                               size=size)

    steps_per_epoch = 0

    if FLAGS.train:
        is_display_step = FLAGS.display_step > 0 and (FLAGS.display_step == 1) 
        num_jobs_train = max(1, model_feeder.train.total_batches / FLAGS.iters_per_worker) * FLAGS.epoch
        steps_per_epoch = max(1, model_feeder.train.total_batches) / hvd.size()
        epoch = Epoch(epoch_id, int(math.ceil( num_jobs_train )), set_name='train', report=is_display_step)

    # Create the optimizer
    # For large batch training, we'll need to do a gradual learning rate warmup and then decay
    if FLAGS.lrn_warmup:
        train_batch_size = FLAGS.train_batch_size * hvd.size()
        lrpw = FLAGS.learning_rate * train_batch_size / 12 # StartingLR * train_batch_size/ original_batch_size
        warmup_steps = FLAGS.warmup_epochs * steps_per_epoch
        lrn_rate = gradual_warmup_then_dec(FLAGS.learning_rate, warmup_steps, lrpw, global_step, num_jobs_train, name="gradual_warmup_then_dec")
        optimizer = create_optimizer(lrn_rate)
    else:
        optimizer = create_optimizer(FLAGS.learning_rate)

    #Using Horovod optimizer
    optimizer = hvd.DistributedOptimizer(optimizer)

    # Get the data_set specific graph end-points
    results_tuple, gradients, mean_edit_distance, loss = get_tower_results(model_feeder, optimizer)

    # Average tower gradients across GPUs
    avg_tower_gradients = average_gradients(gradients)


    # Op to merge all summaries for the summary hook
    merge_all_summaries_op = tf.summary.merge_all()

    # Apply gradients to modify the model
    apply_gradient_op = optimizer.apply_gradients(avg_tower_gradients, global_step=global_step)

    # Collecting the hooks
    hooks = []

    hooks.append(hvd.BroadcastGlobalVariablesHook(0))

    hooks.append(tf.train.StopAtStepHook(last_step=(steps_per_epoch * FLAGS.epoch)))
 
    checkpoint_dir_ = FLAGS.checkpoint_dir if hvd.rank() == 0 else None

    tower_feeder_count = max(len(get_available_gpus()), 1)
    iterators = [model_feeder.iterators[i].initializer for i in range(tower_feeder_count)]
    scaffold = tf.train.Scaffold(local_init_op=tf.group(tf.local_variables_initializer(), iterators))

    if len(FLAGS.initialize_from_frozen_model) > 0:
        with tf.gfile.FastGFile(FLAGS.initialize_from_frozen_model, 'rb') as fin:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(fin.read())

        var_names = [v.name for v in tf.trainable_variables()]
        var_tensors = tf.import_graph_def(graph_def, return_elements=var_names)

        # build a { var_name: var_tensor } dict
        var_tensors = dict(zip(var_names, var_tensors))

        training_graph = tf.get_default_graph()

        assign_ops = []
        for name, restored_tensor in var_tensors.items():
            training_tensor = training_graph.get_tensor_by_name(name)
            assign_ops.append(tf.assign(training_tensor, restored_tensor))

        init_from_frozen_model_op = tf.group(*assign_ops)

    # The MonitoredTrainingSession takes care of session initialization,
    # restoring from a checkpoint, saving to a checkpoint, and closing when done
    # or an error occurs.
    try:
        with tf.train.MonitoredTrainingSession(scaffold=scaffold,
                                                   config = session_config,
                                                   checkpoint_dir=checkpoint_dir_,
                                                   hooks=hooks) as session:
                    if len(FLAGS.initialize_from_frozen_model) > 0:
                        log_info('Initializing from frozen model: {}'.format(FLAGS.initialize_from_frozen_model))
                        session.run(init_from_frozen_model_op)

                    if hvd.rank() == 0:
                        # Retrieving global_step from the (potentially restored) model
                        step = session.run(global_step)
                    
                    job = epoch.get_job(hvd.rank())

                    while job and not session.should_stop():
                        log_debug('Computing %s...' % job)

                        # Initialize loss aggregator
                        total_loss = 0.0

                        # Setting the training operation in case of training requested
                        train_op = apply_gradient_op if job.set_name == 'train' else []

                        # Adding options for collecting trace
                        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        run_metadata = tf.RunMetadata()

                        # Loop over the batches
                        for job_step in range(job.steps):
                            if session.should_stop():
                                break

                            log_debug('Starting batch...')
                            # Compute the batch
                            
                            if not FLAGS.fulltrace:
                                start = time.time()
                                _, current_step, batch_loss = session.run([train_op, global_step, loss])
                                duration = time.time() - start
                            else:
                                # Get the full trace of the run
                                start = time.time()
                                _, current_step, batch_loss, batch_report = session.run([train_op, global_step, loss],options=run_options, run_metadata=run_metadata)
                                duration = time.time() - start
                                if hvd.rank() == 0:
                                    tl = timeline.Timeline(run_metadata.step_stats)
                                    ctf = tl.generate_chrome_trace_format()
                                    with open(FLAGS.trace_filename, 'w') as f:
                                        f.write(ctf)

                            log_debug('Finished batch step %d.' % current_step)
                            if hvd.rank() == 0: 
                               print('Finished batch step:%d on worker:%d' % (current_step, hvd.rank()))
                               print('Training loss:%.2f' % batch_loss)
                               print('Iter duration:%.2f' % duration)
                               print('Iteration end time:%s' % (datetime.datetime.now()))

                            # Add batch to loss
                            total_loss += batch_loss

                            #Get the next training job
                            job = epoch.get_job(hvd.rank())

    except Exception as e:
                    log_error(str(e))
                    traceback.print_exc()
                    # Calling all hook's end() methods to end blocking calls
                    for hook in hooks:
                        hook.end(session)
                    sys.exit(1)

    log_debug('Session closed.')

def create_inference_graph(batch_size=None, use_new_decoder=False):
    # Input tensor will be of shape [batch_size, n_steps, n_input + 2*n_input*n_context]
    input_tensor = tf.placeholder(tf.float32, [batch_size, None, n_input + 2*n_input*n_context], name='input_node')
    seq_length = tf.placeholder(tf.int32, [batch_size], name='input_lengths')

    # Calculate the logits of the batch using BiRNN
    logits = BiRNN(input_tensor, tf.to_int64(seq_length) if FLAGS.use_seq_length else None, no_dropout)

    # Beam search decode the batch
    decoder = decode_with_lm if use_new_decoder else tf.nn.ctc_beam_search_decoder

    decoded, _ = decoder(logits, seq_length, merge_repeated=False, beam_width=FLAGS.beam_width)
    decoded = tf.convert_to_tensor(
        [tf.sparse_tensor_to_dense(sparse_tensor) for sparse_tensor in decoded], name='output_node')

    return (
        {
            'input': input_tensor,
            'input_lengths': seq_length,
        },
        {
            'outputs': decoded,
        }
    )

def do_single_file_inference(input_file_path):
    with tf.Session(config=session_config) as session:
        inputs, outputs = create_inference_graph(batch_size=1, use_new_decoder=True)

        # Create a saver using variables from the above newly created graph
        saver = tf.train.Saver(tf.global_variables())

        # Restore variables from training checkpoint
        # TODO: This restores the most recent checkpoint, but if we use validation to counterract
        #       over-fitting, we may want to restore an earlier checkpoint.
        checkpoint = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if not checkpoint:
            log_error('Checkpoint directory ({}) does not contain a valid checkpoint state.'.format(FLAGS.checkpoint_dir))
            exit(1)

        checkpoint_path = checkpoint.model_checkpoint_path
        saver.restore(session, checkpoint_path)

        mfcc = audiofile_to_input_vector(input_file_path, n_input, n_context)

        output = session.run(outputs['outputs'], feed_dict = {
            inputs['input']: [mfcc],
            inputs['input_lengths']: [len(mfcc)],
        })

        text = ndarray_to_text(output[0][0], alphabet)

        print(text)

def main(_) :

    initialize_globals()

    if FLAGS.train:
        if hvd.rank() == 0:
           print('Start time:%s' % (datetime.datetime.now()))
        train()
        if hvd.rank() == 0:
           print('Training end time:%s' % (datetime.datetime.now()))
        log_debug('Server stopped.')
    else:
        print('The script currently does not support testing or validation')

    if len(FLAGS.one_shot_infer):
        do_single_file_inference(FLAGS.one_shot_infer)

if __name__ == '__main__' :
    tf.app.run()
