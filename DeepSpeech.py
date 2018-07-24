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
import traceback
import inspect

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


# Importer
# ========

tf.app.flags.DEFINE_string  ('train_files',      '',          'comma separated list of files specifying the dataset used for training. multiple files will get merged')
tf.app.flags.DEFINE_string  ('dev_files',        '',          'comma separated list of files specifying the dataset used for validation. multiple files will get merged')
tf.app.flags.DEFINE_string  ('test_files',       '',          'comma separated list of files specifying the dataset used for testing. multiple files will get merged')
tf.app.flags.DEFINE_boolean ('fulltrace',        False,       'if full trace debug info should be generated during training')

# Cluster configuration
# =====================

tf.app.flags.DEFINE_string  ('ps_hosts',         '',          'parameter servers - comma separated list of hostname:port pairs')
tf.app.flags.DEFINE_string  ('worker_hosts',     '',          'workers - comma separated list of hostname:port pairs')
tf.app.flags.DEFINE_string  ('job_name',         'localhost', 'job name - one of localhost (default), worker, ps')
tf.app.flags.DEFINE_integer ('task_index',       0,           'index of task within the job - worker with index 0 will be the chief')
tf.app.flags.DEFINE_integer ('replicas',         -1,          'total number of replicas - if negative, its absolute value is multiplied by the number of workers')
tf.app.flags.DEFINE_integer ('replicas_to_agg',  -1,          'number of replicas to aggregate - if negative, its absolute value is multiplied by the number of workers')
tf.app.flags.DEFINE_integer ('coord_retries',    100,         'number of tries of workers connecting to training coordinator before failing')
tf.app.flags.DEFINE_string  ('coord_host',       'localhost', 'coordination server host')
tf.app.flags.DEFINE_integer ('coord_port',       2500,        'coordination server port')
tf.app.flags.DEFINE_integer ('iters_per_worker', 1,           'number of train or inference iterations per worker before results are sent back to coordinator')

# Global Constants
# ================

tf.app.flags.DEFINE_boolean ('train',            True,        'whether to train the network')
tf.app.flags.DEFINE_boolean ('test',             True,        'whether to test the network')
tf.app.flags.DEFINE_integer ('epoch',            75,          'target epoch to train - if negative, the absolute number of additional epochs will be trained')

tf.app.flags.DEFINE_boolean ('use_warpctc',      False,       'whether to use GPU bound Warp-CTC')

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

# Batch sizes

tf.app.flags.DEFINE_integer ('train_batch_size', 1,           'number of elements in a training batch')
tf.app.flags.DEFINE_integer ('dev_batch_size',   1,           'number of elements in a validation batch')
tf.app.flags.DEFINE_integer ('test_batch_size',  1,           'number of elements in a test batch')

# Sample limits

tf.app.flags.DEFINE_integer ('limit_train',      0,           'maximum number of elements to use from train set - 0 means no limit')
tf.app.flags.DEFINE_integer ('limit_dev',        0,           'maximum number of elements to use from validation set- 0 means no limit')
tf.app.flags.DEFINE_integer ('limit_test',       0,           'maximum number of elements to use from test set- 0 means no limit')

# Step widths

tf.app.flags.DEFINE_integer ('display_step',     0,           'number of epochs we cycle through before displaying detailed progress - 0 means no progress display')
tf.app.flags.DEFINE_integer ('validation_step',  0,           'number of epochs we cycle through before validating the model - a detailed progress report is dependent on "--display_step" - 0 means no validation steps')

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

tf.app.flags.DEFINE_integer ('log_level',        1,           'log level for console logs - 0: INFO, 1: WARN, 2: ERROR, 3: FATAL')
tf.app.flags.DEFINE_boolean ('log_traffic',      False,       'log cluster transaction and traffic information during debug logging')

tf.app.flags.DEFINE_string  ('wer_log_pattern',  '',          'pattern for machine readable global logging of WER progress; has to contain %%s, %%s and %%f for the set name, the date and the float respectively; example: "GLOBAL LOG: logwer(\'12ade231\', %%s, %%s, %%f)" would result in some entry like "GLOBAL LOG: logwer(\'12ade231\', \'train\', \'2017-05-18T03:09:48-0700\', 0.05)"; if omitted (default), there will be no logging')

tf.app.flags.DEFINE_boolean ('log_placement',    False,       'whether to log device placement of the operators to the console')
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

# This parameter is irrespective of the time taken by single epoch to complete and checkpoint saving intervals.
# It is possible that early stopping is triggered far after the best checkpoint is already replaced by checkpoint saving interval mechanism.
# One has to align the parameters (earlystop_nsteps, checkpoint_secs) accordingly as per the time taken by an epoch on different datasets.

tf.app.flags.DEFINE_integer ('earlystop_nsteps',  4,          'number of steps to consider for early stopping. Loss is not stored in the checkpoint so when checkpoint is revived it starts the loss calculation from start at that point')
tf.app.flags.DEFINE_float   ('estop_mean_thresh', 0.5,        'mean threshold for loss to determine the condition if early stopping is required')
tf.app.flags.DEFINE_float   ('estop_std_thresh',  0.5,        'standard deviation threshold for loss to determine the condition if early stopping is required')

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

for var in ['b1', 'h1', 'b2', 'h2', 'b3', 'h3', 'b5', 'h5', 'b6', 'h6']:
    tf.app.flags.DEFINE_float('%s_stddev' % var, None, 'standard deviation to use when initialising %s' % var)

FLAGS = tf.app.flags.FLAGS

def initialize_globals():

    # ps and worker hosts required for p2p cluster setup
    FLAGS.ps_hosts = list(filter(len, FLAGS.ps_hosts.split(',')))
    FLAGS.worker_hosts = list(filter(len, FLAGS.worker_hosts.split(',')))

    # Determine, if we are the chief worker
    global is_chief
    is_chief = len(FLAGS.worker_hosts) == 0 or (FLAGS.task_index == 0 and FLAGS.job_name == 'worker')

    # Initializing and starting the training coordinator
    global COORD
    COORD = TrainingCoordinator()
    COORD.start()

    # The absolute number of computing nodes - regardless of cluster or single mode
    global num_workers
    num_workers = max(1, len(FLAGS.worker_hosts))

    # Create a cluster from the parameter server and worker hosts.
    global cluster
    cluster = tf.train.ClusterSpec({'ps': FLAGS.ps_hosts, 'worker': FLAGS.worker_hosts})

    # If replica numbers are negative, we multiply their absolute values with the number of workers
    if FLAGS.replicas < 0:
        FLAGS.replicas = num_workers * -FLAGS.replicas
    if FLAGS.replicas_to_agg < 0:
        FLAGS.replicas_to_agg = num_workers * -FLAGS.replicas_to_agg

    # The device path base for this node
    global worker_device
    worker_device = '/job:%s/task:%d' % (FLAGS.job_name, FLAGS.task_index)

    # This node's CPU device
    global cpu_device
    cpu_device = worker_device + '/cpu:0'

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
    dropout_rates = [tf.placeholder(tf.float32, name='dropout_{}'.format(i)) for i in range(6)]

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
    session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=FLAGS.log_placement)

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

    # Queues that are used to gracefully stop parameter servers.
    # Each queue stands for one ps. A finishing worker sends a token to each queue before joining/quitting.
    # Each ps will dequeue as many tokens as there are workers before joining/quitting.
    # This ensures parameter servers won't quit, if still required by at least one worker and
    # also won't wait forever (like with a standard `server.join()`).
    global done_queues
    done_queues = []
    for i, ps in enumerate(FLAGS.ps_hosts):
        # Queues are hosted by their respective owners
        with tf.device('/job:ps/task:%d' % i):
            done_queues.append(tf.FIFOQueue(1, tf.int32, shared_name=('queue%i' % i)))

    # Placeholder to pass in the worker's index as token
    global token_placeholder
    token_placeholder = tf.placeholder(tf.int32)

    # Enqueue operations for each parameter server
    global done_enqueues
    done_enqueues = [queue.enqueue(token_placeholder) for queue in done_queues]

    # Dequeue operations for each parameter server
    global done_dequeues
    done_dequeues = [queue.dequeue() for queue in done_queues]

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
    # Use the /cpu:0 device on worker_device for scoped operations
    if len(FLAGS.ps_hosts) == 0:
        device = worker_device
    else:
        device = tf.train.replica_device_setter(worker_device=worker_device, cluster=cluster)

    with tf.device(device):
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

def calculate_mean_edit_distance_and_loss(model_feeder, tower, dropout):
    r'''
    This routine beam search decodes a mini-batch and calculates the loss and mean edit distance.
    Next to total and average loss it returns the mean edit distance,
    the decoded result and the batch's original Y.
    '''
    # Obtain the next batch of data
    batch_x, batch_seq_len, batch_y = model_feeder.next_batch(tower)

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
def create_optimizer():
    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate,
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
            if len(FLAGS.ps_hosts) == 0:
                device = available_devices[i]
            else:
                device = tf.train.replica_device_setter(worker_device=available_devices[i], cluster=cluster)
            with tf.device(device):
                # Create a scope for all operations of tower i
                with tf.name_scope('tower_%d' % i) as scope:
                    # Calculate the avg_loss and mean_edit_distance and retrieve the decoded
                    # batch along with the original batch's labels (Y) of this tower
                    total_loss, avg_loss, distance, mean_edit_distance, decoded, labels = \
                        calculate_mean_edit_distance_and_loss(model_feeder, i, dropout_rates)

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

    avg_loss_across_towers = tf.reduce_mean(tower_avg_losses, 0)

    tf.summary.scalar(name='step_loss', tensor=avg_loss_across_towers, collections=['step_summaries'])

    # Return the results tuple, the gradients, and the means of mean edit distances and losses
    return (tower_labels, tower_decodings, tower_distances, tower_total_losses), \
           tower_gradients, \
           tf.reduce_mean(tower_mean_edit_distances, 0), \
           avg_loss_across_towers


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


# Helpers
# =======

def calculate_report(results_tuple):
    r'''
    This routine will calculate a WER report.
    It'll compute the `mean` WER and create ``Sample`` objects of the ``report_count`` top lowest
    loss items from the provided WER results tuple (only items with WER!=0 and ordered by their WER).
    '''
    samples = []
    items = list(zip(*results_tuple))
    total_levenshtein = 0.0
    total_label_length = 0.0
    for label, decoding, distance, loss in items:
        sample_wer = wer(label, decoding)
        sample = Sample(label, decoding, loss, distance, sample_wer)
        samples.append(sample)
        total_levenshtein += levenshtein(label.split(), decoding.split())
        total_label_length += float(len(label.split()))

    # Getting the WER from the accumulated levenshteins and lengths
    samples_wer = total_levenshtein / total_label_length

    # Filter out all items with WER=0
    samples = [s for s in samples if s.wer > 0]

    # Order the remaining items by their loss (lowest loss on top)
    samples.sort(key=lambda s: s.loss)

    # Take only the first report_count items
    samples = samples[:FLAGS.report_count]

    # Order this top FLAGS.report_count items by their WER (lowest WER on top)
    samples.sort(key=lambda s: s.wer)

    return samples_wer, samples

def collect_results(results_tuple, returns):
    r'''
    This routine will help collecting partial results for the WER reports.
    The ``results_tuple`` is composed of an array of the original labels,
    an array of the corresponding decodings, an array of the corrsponding
    distances and an array of the corresponding losses. ``returns`` is built up
    in a similar way, containing just the unprocessed results of one
    ``session.run`` call (effectively of one batch).
    Labels and decodings are converted to text before splicing them into their
    corresponding results_tuple lists. In the case of decodings,
    for now we just pick the first available path.
    '''
    # Each of the arrays within results_tuple will get extended by a batch of each available device
    for i in range(len(available_devices)):
        # Collect the labels
        results_tuple[0].extend(sparse_tensor_value_to_texts(returns[0][i], alphabet))

        # Collect the decodings - at the moment we default to the first one
        results_tuple[1].extend(sparse_tensor_value_to_texts(returns[1][i][0], alphabet))

        # Collect the distances
        results_tuple[2].extend(returns[2][i])

        # Collect the losses
        results_tuple[3].extend(returns[3][i])


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

    def finish_job(self, job):
        '''Finishes a running job. Removes it from the running jobs list and adds it to the done jobs list.

        Args:
            job (WorkerJob): the job to put into state 'done'
        '''
        index = next((i for i in range(len(self.jobs_running)) if self.jobs_running[i].id == job.id), -1)
        if index >= 0:
            self.jobs_running.pop(index)
            self.jobs_done.append(job)
            log_traffic('%s - Moved %s from running to done.' % (self.name(), job))
        else:
            log_warn('%s - There is no job with ID %d registered as running.' % (self.name(), job.id))

    def done(self):
        '''Checks, if all jobs of the epoch are in state 'done'.
        It also lazy-prepares a WER report from the result data of all jobs.

        Returns:
            bool. if all jobs of the epoch are 'done'
        '''
        if len(self.jobs_open) == 0 and len(self.jobs_running) == 0:
            num_jobs = len(self.jobs_done)
            if num_jobs > 0:
                jobs = self.jobs_done
                self.jobs_done = []
                if not self.num_jobs == num_jobs:
                    log_warn('%s - Number of steps not equal to number of jobs done.' % (self.name()))

                agg_loss = 0.0
                agg_wer = 0.0
                agg_mean_edit_distance = 0.0

                for i in range(num_jobs):
                    job = jobs.pop(0)
                    agg_loss += job.loss
                    if self.report:
                        agg_wer += job.wer
                        agg_mean_edit_distance += job.mean_edit_distance
                        self.samples.extend(job.samples)

                self.loss = agg_loss / num_jobs

                # if the job was for validation dataset then append it to the COORD's _loss for early stop verification
                if (FLAGS.early_stop is True) and (self.set_name == 'dev'):
                    COORD._dev_losses.append(self.loss)

                if self.report:
                    self.wer = agg_wer / num_jobs
                    self.mean_edit_distance = agg_mean_edit_distance / num_jobs

                    # Order samles by their loss (lowest loss on top)
                    self.samples.sort(key=lambda s: s.loss)

                    # Take only the first report_count items
                    self.samples = self.samples[:FLAGS.report_count]

                    # Order this top FLAGS.report_count items by their WER (lowest WER on top)
                    self.samples.sort(key=lambda s: s.wer)

                    # Append WER to WER log file
                    if len(FLAGS.wer_log_pattern) > 0:
                        time = datetime.datetime.utcnow().isoformat()
                        # Log WER progress
                        print(FLAGS.wer_log_pattern % (time, self.set_name, self.wer))

            return True
        return False

    def job_status(self):
        '''Provides a printable overview of the states of the jobs of this epoch.

        Returns:
            str. printable overall job state
        '''
        return '%s - jobs open: %d, jobs running: %d, jobs done: %d' % (self.name(), len(self.jobs_open), len(self.jobs_running), len(self.jobs_done))

    def __str__(self):
        if not self.done():
            return self.job_status()

        if not self.report:
            return '%s - loss: %f' % (self.name(), self.loss)

        s = '%s - WER: %f, loss: %s, mean edit distance: %f' % (self.name(), self.wer, self.loss, self.mean_edit_distance)
        if len(self.samples) > 0:
            line = '\n' + ('-' * 80)
            for sample in self.samples:
                s += '%s\n%s' % (line, sample)
            s += line
        return s


class TrainingCoordinator(object):
    class TrainingCoordinationHandler(BaseHTTPServer.BaseHTTPRequestHandler):
        '''Handles HTTP requests from remote workers to the Training Coordinator.
        '''
        def _send_answer(self, data=None):
            self.send_response(200)
            self.send_header('content-type', 'text/plain')
            self.end_headers()
            if data:
                self.wfile.write(data)

        def do_GET(self):
            if COORD.started:
                if self.path.startswith(PREFIX_NEXT_INDEX):
                    index = COORD.get_next_index(self.path[len(PREFIX_NEXT_INDEX):])
                    if index >= 0:
                        self._send_answer(str(index).encode("utf-8"))
                        return
                elif self.path.startswith(PREFIX_GET_JOB):
                    job = COORD.get_job(worker=int(self.path[len(PREFIX_GET_JOB):]))
                    if job:
                        self._send_answer(pickle.dumps(job))
                        return
                self.send_response(204) # end of training
            else:
                self.send_response(202) # not ready yet
            self.end_headers()

        def do_POST(self):
            if COORD.started:
                src = self.rfile.read(int(self.headers['content-length']))
                job = COORD.next_job(pickle.loads(src))
                if job:
                    self._send_answer(pickle.dumps(job))
                    return
                self.send_response(204) # end of training
            else:
                self.send_response(202) # not ready yet
            self.end_headers()

        def log_message(self, format, *args):
            '''Overriding base method to suppress web handler messages on stdout.
            '''
            return


    def __init__(self):
        ''' Central training coordination class.
        Used for distributing jobs among workers of a cluster.
        Instantiated on all workers, calls of non-chief workers will transparently
        HTTP-forwarded to the chief worker instance.
        '''
        self._init()
        self._lock = Lock()
        self.started = False
        if is_chief:
            self._httpd = BaseHTTPServer.HTTPServer((FLAGS.coord_host, FLAGS.coord_port), TrainingCoordinator.TrainingCoordinationHandler)

    def _reset_counters(self):
        self._index_train = 0
        self._index_dev = 0
        self._index_test = 0

    def _init(self):
        self._epochs_running = []
        self._epochs_done = []
        self._reset_counters()
        self._dev_losses = []

    def _log_all_jobs(self):
        '''Use this to debug-print epoch state'''
        log_debug('Epochs - running: %d, done: %d' % (len(self._epochs_running), len(self._epochs_done)))
        for epoch in self._epochs_running:
            log_debug('       - running: ' + epoch.job_status())

    def start_coordination(self, model_feeder, step=0):
        '''Starts to coordinate epochs and jobs among workers on base of
        data-set sizes, the (global) step and FLAGS parameters.

        Args:
            model_feeder (ModelFeeder): data-sets to be used for coordinated training

        Kwargs:
            step (int): global step of a loaded model to determine starting point
        '''
        with self._lock:
            self._init()

            # Number of GPUs per worker - fixed for now by local reality or cluster setup
            gpus_per_worker = len(available_devices)

            # Number of batches processed per job per worker
            batches_per_job  = gpus_per_worker * max(1, FLAGS.iters_per_worker)

            # Number of batches per global step
            batches_per_step = gpus_per_worker * max(1, FLAGS.replicas_to_agg)

            # Number of global steps per epoch - to be at least 1
            steps_per_epoch = max(1, model_feeder.train.total_batches // batches_per_step)

            # The start epoch of our training
            self._epoch = step // steps_per_epoch

            # Number of additional 'jobs' trained already 'on top of' our start epoch
            jobs_trained = (step % steps_per_epoch) * batches_per_step // batches_per_job

            # Total number of train/dev/test jobs covering their respective whole sets (one epoch)
            self._num_jobs_train = max(1, model_feeder.train.total_batches // batches_per_job)
            self._num_jobs_dev   = max(1, model_feeder.dev.total_batches   // batches_per_job)
            self._num_jobs_test  = max(1, model_feeder.test.total_batches  // batches_per_job)

            if FLAGS.epoch < 0:
                # A negative epoch means to add its absolute number to the epochs already computed
                self._target_epoch = self._epoch + abs(FLAGS.epoch)
            else:
                self._target_epoch = FLAGS.epoch

            # State variables
            # We only have to train, if we are told so and are not at the target epoch yet
            self._train = FLAGS.train and self._target_epoch > self._epoch
            self._test = FLAGS.test

            if self._train:
                # The total number of jobs for all additional epochs to be trained
                # Will be decremented for each job that is produced/put into state 'open'
                self._num_jobs_train_left = (self._target_epoch - self._epoch) * self._num_jobs_train - jobs_trained
                log_info('STARTING Optimization')
                self._training_time = stopwatch()

            # Important for debugging
            log_debug('step: %d' % step)
            log_debug('epoch: %d' % self._epoch)
            log_debug('target epoch: %d' % self._target_epoch)
            log_debug('steps per epoch: %d' % steps_per_epoch)
            log_debug('number of batches in train set: %d' % model_feeder.train.total_batches)
            log_debug('batches per job: %d' % batches_per_job)
            log_debug('batches per step: %d' % batches_per_step)
            log_debug('number of jobs in train set: %d' % self._num_jobs_train)
            log_debug('number of jobs already trained in first epoch: %d' % jobs_trained)

            self._next_epoch()

        # The coordinator is ready to serve
        self.started = True

    def _next_epoch(self):
        # State-machine of the coordination process

        # Indicates, if there were 'new' epoch(s) provided
        result = False

        # Make sure that early stop is enabled and validation part is enabled
        if (FLAGS.early_stop is True) and (FLAGS.validation_step > 0) and (len(self._dev_losses) >= FLAGS.earlystop_nsteps):

            # Calculate the mean of losses for past epochs
            mean_loss = np.mean(self._dev_losses[-FLAGS.earlystop_nsteps:-1])
            # Calculate the standard deviation for losses from validation part in the past epochs
            std_loss = np.std(self._dev_losses[-FLAGS.earlystop_nsteps:-1])
            # Update the list of losses incurred
            self._dev_losses = self._dev_losses[-FLAGS.earlystop_nsteps:]
            log_debug('Checking for early stopping (last %d steps) validation loss: %f, with standard deviation: %f and mean: %f' % (FLAGS.earlystop_nsteps, self._dev_losses[-1], std_loss, mean_loss))

            # Check if validation loss has started increasing or is not decreasing substantially, making sure slight fluctuations don't bother the early stopping from working
            if self._dev_losses[-1] > np.max(self._dev_losses[:-1]) or (abs(self._dev_losses[-1] - mean_loss) < FLAGS.estop_mean_thresh and std_loss < FLAGS.estop_std_thresh):
                # Time to early stop
                log_info('Early stop triggered as (for last %d steps) validation loss: %f with standard deviation: %f and mean: %f' % (FLAGS.earlystop_nsteps, self._dev_losses[-1], std_loss, mean_loss))
                self._dev_losses = []
                self._end_training()
                self._train = False

        if self._train:
            # We are in train mode
            if self._num_jobs_train_left > 0:
                # There are still jobs left
                num_jobs_train = min(self._num_jobs_train_left, self._num_jobs_train)
                self._num_jobs_train_left -= num_jobs_train

                # Let's try our best to keep the notion of curriculum learning
                self._reset_counters()

                # If the training part of the current epoch should generate a WER report
                is_display_step = FLAGS.display_step > 0 and (FLAGS.display_step == 1 or self._epoch > 0) and (self._epoch % FLAGS.display_step == 0 or self._epoch == self._target_epoch)
                # Append the training epoch
                self._epochs_running.append(Epoch(self._epoch, num_jobs_train, set_name='train', report=is_display_step))

                if FLAGS.validation_step > 0 and (FLAGS.validation_step == 1 or self._epoch > 0) and self._epoch % FLAGS.validation_step == 0:
                    # The current epoch should also have a validation part
                    self._epochs_running.append(Epoch(self._epoch, self._num_jobs_dev, set_name='dev', report=is_display_step))


                # Indicating that there were 'new' epoch(s) provided
                result = True
            else:
                # No jobs left, but still in train mode: concluding training
                self._end_training()
                self._train = False

        if self._test and not self._train:
            # We shall test, and are not in train mode anymore
            self._test = False
            self._epochs_running.append(Epoch(self._epoch, self._num_jobs_test, set_name='test', report=True))
            # Indicating that there were 'new' epoch(s) provided
            result = True

        if result:
            # Increment the epoch index - shared among train and test 'state'
            self._epoch += 1
        return result

    def _end_training(self):
        self._training_time = stopwatch(self._training_time)
        log_info('FINISHED Optimization - training time: %s' % format_duration(self._training_time))

    def start(self):
        '''Starts Training Coordinator. If chief, it starts a web server for
        communication with non-chief instances.
        '''
        if is_chief:
            log_debug('Starting coordinator...')
            self._thread = Thread(target=self._httpd.serve_forever)
            self._thread.daemon = True
            self._thread.start()
            log_debug('Coordinator started.')

    def stop(self, wait_for_running_epochs=True):
        '''Stops Training Coordinator. If chief, it waits for all epochs to be
        'done' and then shuts down the web server.
        '''
        if is_chief:
            if wait_for_running_epochs:
                while len(self._epochs_running) > 0:
                    log_traffic('Coordinator is waiting for epochs to finish...')
                    time.sleep(5)
            log_debug('Stopping coordinator...')
            self._httpd.shutdown()
            log_debug('Coordinator stopped.')

    def _talk_to_chief(self, path, data=None, default=None):
        tries = 0
        while tries < FLAGS.coord_retries:
            tries += 1
            try:
                url = 'http://%s:%d%s' % (FLAGS.coord_host, FLAGS.coord_port, path)
                log_traffic('Contacting coordinator - url: %s, tries: %d ...' % (url, tries-1))
                res = urllib.request.urlopen(urllib.request.Request(url, data, { 'content-type': 'text/plain' }))
                str = res.read()
                status = res.getcode()
                log_traffic('Coordinator responded - url: %s, status: %s' % (url, status))
                if status == 200:
                    return str
                if status == 204: # We use 204 (no content) to indicate end of training
                    return default
            except urllib.error.HTTPError as error:
                log_traffic('Problem reaching coordinator - url: %s, HTTP code: %d' % (url, error.code))
                pass
            time.sleep(10)
        return default

    def get_next_index(self, set_name):
        '''Retrives a new cluster-unique batch index for a given set-name.
        Prevents applying one batch multiple times per epoch.

        Args:
            set_name (str): name of the data set - one of 'train', 'dev', 'test'

        Returns:
            int. new data set index
        '''
        with self._lock:
            if is_chief:
                member = '_index_' + set_name
                value = getattr(self, member, -1)
                setattr(self, member, value + 1)
                return value
            else:
                # We are a remote worker and have to hand over to the chief worker by HTTP
                log_traffic('Asking for next index...')
                value = int(self._talk_to_chief(PREFIX_NEXT_INDEX + set_name))
                log_traffic('Got index %d.' % value)
                return value

    def _get_job(self, worker=0):
        job = None
        # Find first running epoch that provides a next job
        for epoch in self._epochs_running:
            job = epoch.get_job(worker)
            if job:
                return job
        # No next job found
        return None

    def get_job(self, worker=0):
        '''Retrieves the first job for a worker.

        Kwargs:
            worker (int): index of the worker to get the first job for

        Returns:
            WorkerJob. a job of one of the running epochs that will get
                associated with the given worker and put into state 'running'
        '''
        # Let's ensure that this does not interfere with other workers/requests
        with self._lock:
            if is_chief:
                # First try to get a next job
                job = self._get_job(worker)

                if job is None:
                    # If there was no next job, we give it a second chance by triggering the epoch state machine
                    if self._next_epoch():
                        # Epoch state machine got a new epoch
                        # Second try to get a next job
                        job = self._get_job(worker)
                        if job is None:
                            # Albeit the epoch state machine got a new epoch, the epoch had no new job for us
                            log_error('Unexpected case - no job for worker %d.' % (worker))
                        return job

                    # Epoch state machine has no new epoch
                    # This happens at the end of the whole training - nothing to worry about
                    log_traffic('No jobs left for worker %d.' % (worker))
                    self._log_all_jobs()
                    return None

                # We got a new job from one of the currently running epochs
                log_traffic('Got new %s' % job)
                return job

            # We are a remote worker and have to hand over to the chief worker by HTTP
            result = self._talk_to_chief(PREFIX_GET_JOB + str(FLAGS.task_index))
            if result:
                result = pickle.loads(result)
            return result

    def next_job(self, job):
        '''Sends a finished job back to the coordinator and retrieves in exchange the next one.

        Kwargs:
            job (WorkerJob): job that was finished by a worker and who's results are to be
                digested by the coordinator

        Returns:
            WorkerJob. next job of one of the running epochs that will get
                associated with the worker from the finished job and put into state 'running'
        '''
        if is_chief:
            # Try to find the epoch the job belongs to
            epoch = next((epoch for epoch in self._epochs_running if epoch.id == job.epoch_id), None)
            if epoch:
                # We are going to manipulate things - let's avoid undefined state
                with self._lock:
                    # Let the epoch finish the job
                    epoch.finish_job(job)
                    # Check, if epoch is done now
                    if epoch.done():
                        # If it declares itself done, move it from 'running' to 'done' collection
                        self._epochs_running.remove(epoch)
                        self._epochs_done.append(epoch)
                        log_info('%s' % epoch)
            else:
                # There was no running epoch found for this job - this should never happen.
                log_error('There is no running epoch of ID %d for job with ID %d. This should never happen.' % (job.epoch_id, job.id))
            return self.get_job(job.worker)

        # We are a remote worker and have to hand over to the chief worker by HTTP
        result = self._talk_to_chief('', data=pickle.dumps(job))
        if result:
            result = pickle.loads(result)
        return result

def send_token_to_ps(session, kill=False):
    # Sending our token (the task_index as a debug opportunity) to each parameter server.
    # kill switch tokens are negative and decremented by 1 to deal with task_index 0
    token = -FLAGS.task_index-1 if kill else FLAGS.task_index
    kind = 'kill switch' if kill else 'stop'
    for index, enqueue in enumerate(done_enqueues):
        log_debug('Sending %s token to ps %d...' % (kind, index))
        session.run(enqueue, feed_dict={ token_placeholder: token })
        log_debug('Sent %s token to ps %d.' % (kind, index))

def train(server=None):
    r'''
    Trains the network on a given server of a cluster.
    If no server provided, it performs single process training.
    '''

    # Create a variable to hold the global_step.
    # It will automagically get incremented by the optimizer.
    global_step = tf.Variable(0, trainable=False, name='global_step')

    # Reading training set
    train_set = DataSet(FLAGS.train_files.split(','),
                        FLAGS.train_batch_size,
                        limit=FLAGS.limit_train,
                        next_index=lambda i: COORD.get_next_index('train'))

    # Reading validation set
    dev_set = DataSet(FLAGS.dev_files.split(','),
                      FLAGS.dev_batch_size,
                      limit=FLAGS.limit_dev,
                      next_index=lambda i: COORD.get_next_index('dev'))

    # Reading test set
    test_set = DataSet(FLAGS.test_files.split(','),
                       FLAGS.test_batch_size,
                       limit=FLAGS.limit_test,
                       next_index=lambda i: COORD.get_next_index('test'))

    # Combining all sets to a multi set model feeder
    model_feeder = ModelFeeder(train_set,
                               dev_set,
                               test_set,
                               n_input,
                               n_context,
                               alphabet,
                               tower_feeder_count=len(available_devices))

    # Create the optimizer
    optimizer = create_optimizer()

    # Synchronous distributed training is facilitated by a special proxy-optimizer
    if not server is None:
        optimizer = tf.train.SyncReplicasOptimizer(optimizer,
                                                   replicas_to_aggregate=FLAGS.replicas_to_agg,
                                                   total_num_replicas=FLAGS.replicas)

    # Get the data_set specific graph end-points
    results_tuple, gradients, mean_edit_distance, loss = get_tower_results(model_feeder, optimizer)

    # Average tower gradients across GPUs
    avg_tower_gradients = average_gradients(gradients)

    # Add summaries of all variables and gradients to log
    log_grads_and_vars(avg_tower_gradients)

    # Op to merge all summaries for the summary hook
    merge_all_summaries_op = tf.summary.merge_all()

    # These are saved on every step
    step_summaries_op = tf.summary.merge_all('step_summaries')

    step_summary_writers = {
        'train': tf.summary.FileWriter(os.path.join(FLAGS.summary_dir, 'train'), max_queue=120),
        'dev': tf.summary.FileWriter(os.path.join(FLAGS.summary_dir, 'dev'), max_queue=120),
        'test': tf.summary.FileWriter(os.path.join(FLAGS.summary_dir, 'test'), max_queue=120)
    }

    # Apply gradients to modify the model
    apply_gradient_op = optimizer.apply_gradients(avg_tower_gradients, global_step=global_step)


    if FLAGS.early_stop is True and not FLAGS.validation_step > 0:
        log_warn('Parameter --validation_step needs to be >0 for early stopping to work')

    class CoordHook(tf.train.SessionRunHook):
        r'''
        Embedded coordination hook-class that will use variables of the
        surrounding Python context.
        '''
        def after_create_session(self, session, coord):
            log_debug('Starting queue runners...')
            model_feeder.start_queue_threads(session, coord)
            log_debug('Queue runners started.')

        def end(self, session):
            # Closing the data_set queues
            log_debug('Closing queues...')
            model_feeder.close_queues(session)
            log_debug('Queues closed.')

            # Telling the ps that we are done
            send_token_to_ps(session)

    # Collecting the hooks
    hooks = [CoordHook()]

    # Hook to handle initialization and queues for sync replicas.
    if not server is None:
        hooks.append(optimizer.make_session_run_hook(is_chief))

    # Hook to save TensorBoard summaries
    if FLAGS.summary_secs > 0:
        hooks.append(tf.train.SummarySaverHook(save_secs=FLAGS.summary_secs, output_dir=FLAGS.summary_dir, summary_op=merge_all_summaries_op))

    # Hook wih number of checkpoint files to save in checkpoint_dir
    if FLAGS.train and FLAGS.max_to_keep > 0:
        saver = tf.train.Saver(max_to_keep=FLAGS.max_to_keep)
        hooks.append(tf.train.CheckpointSaverHook(checkpoint_dir=FLAGS.checkpoint_dir, save_secs=FLAGS.checkpoint_secs, saver=saver))

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

    no_dropout_feed_dict = {
        dropout_rates[0]: 0.,
        dropout_rates[1]: 0.,
        dropout_rates[2]: 0.,
        dropout_rates[3]: 0.,
        dropout_rates[4]: 0.,
        dropout_rates[5]: 0.,
    }

    # The MonitoredTrainingSession takes care of session initialization,
    # restoring from a checkpoint, saving to a checkpoint, and closing when done
    # or an error occurs.
    try:
        with tf.train.MonitoredTrainingSession(master='' if server is None else server.target,
                                               is_chief=is_chief,
                                               hooks=hooks,
                                               checkpoint_dir=FLAGS.checkpoint_dir,
                                               save_checkpoint_secs=None, # already taken care of by a hook
                                               config=session_config) as session:
            if len(FLAGS.initialize_from_frozen_model) > 0:
                log_info('Initializing from frozen model: {}'.format(FLAGS.initialize_from_frozen_model))
                model_feeder.set_data_set(no_dropout_feed_dict, model_feeder.train)
                session.run(init_from_frozen_model_op, feed_dict=no_dropout_feed_dict)

            try:
                if is_chief:
                    # Retrieving global_step from the (potentially restored) model
                    model_feeder.set_data_set(no_dropout_feed_dict, model_feeder.train)
                    step = session.run(global_step, feed_dict=no_dropout_feed_dict)
                    COORD.start_coordination(model_feeder, step)

                # Get the first job
                job = COORD.get_job()

                while job and not session.should_stop():
                    log_debug('Computing %s...' % job)

                    is_train = job.set_name == 'train'

                    # The feed_dict (mainly for switching between queues)
                    if is_train:
                        feed_dict = {
                            dropout_rates[0]: FLAGS.dropout_rate,
                            dropout_rates[1]: FLAGS.dropout_rate2,
                            dropout_rates[2]: FLAGS.dropout_rate3,
                            dropout_rates[3]: FLAGS.dropout_rate4,
                            dropout_rates[4]: FLAGS.dropout_rate5,
                            dropout_rates[5]: FLAGS.dropout_rate6,
                        }
                    else:
                        feed_dict = no_dropout_feed_dict

                    # Sets the current data_set for the respective placeholder in feed_dict
                    model_feeder.set_data_set(feed_dict, getattr(model_feeder, job.set_name))

                    # Initialize loss aggregator
                    total_loss = 0.0

                    # Setting the training operation in case of training requested
                    train_op = apply_gradient_op if is_train else []

                    # Requirements to display a WER report
                    if job.report:
                        # Reset mean edit distance
                        total_mean_edit_distance = 0.0
                        # Create report results tuple
                        report_results = ([],[],[],[])
                        # Extend the session.run parameters
                        report_params = [results_tuple, mean_edit_distance]
                    else:
                        report_params = []

                    # So far the only extra parameter is the feed_dict
                    extra_params = { 'feed_dict': feed_dict }

                    step_summary_writer = step_summary_writers.get(job.set_name)

                    # Loop over the batches
                    for job_step in range(job.steps):
                        if session.should_stop():
                            break

                        log_debug('Starting batch...')
                        # Compute the batch
                        _, current_step, batch_loss, batch_report, step_summary = session.run([train_op, global_step, loss, report_params, step_summaries_op], **extra_params)

                        # Log step summaries
                        step_summary_writer.add_summary(step_summary, current_step)

                        # Uncomment the next line for debugging race conditions / distributed TF
                        log_debug('Finished batch step %d.' % current_step)

                        # Add batch to loss
                        total_loss += batch_loss

                        if job.report:
                            # Collect individual sample results
                            collect_results(report_results, batch_report[0])
                            # Add batch to total_mean_edit_distance
                            total_mean_edit_distance += batch_report[1]

                    # Gathering job results
                    job.loss = total_loss / job.steps
                    if job.report:
                        job.mean_edit_distance = total_mean_edit_distance / job.steps
                        job.wer, job.samples = calculate_report(report_results)


                    # Send the current job to coordinator and receive the next one
                    log_debug('Sending %s...' % job)
                    job = COORD.next_job(job)
            except Exception as e:
                log_error(str(e))
                traceback.print_exc()
                # Calling all hook's end() methods to end blocking calls
                for hook in hooks:
                    hook.end(session)
                # Only chief has a SyncReplicasOptimizer queue runner that needs to be stopped for unblocking process exit.
                # A rather graceful way to do this is by stopping the ps.
                # Only one party can send it w/o failing.
                if is_chief:
                    send_token_to_ps(session, kill=True)
                sys.exit(1)

        log_debug('Session closed.')

    except tf.errors.InvalidArgumentError as e:
        log_error(str(e))
        log_error('The checkpoint in {0} does not match the shapes of the model.'
                  ' Did you change alphabet.txt or the --n_hidden parameter'
                  ' between train runs using the same checkpoint dir? Try moving'
                  ' or removing the contents of {0}.'.format(FLAGS.checkpoint_dir))
        sys.exit(1)

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


def export():
    r'''
    Restores the trained variables into a simpler graph that will be exported for serving.
    '''
    log_info('Exporting the model...')
    with tf.device('/cpu:0'):

        tf.reset_default_graph()
        session = tf.Session(config=session_config)

        inputs, outputs = create_inference_graph()

        # TODO: Transform the decoded output to a string

        # Create a saver and exporter using variables from the above newly created graph
        saver = tf.train.Saver(tf.global_variables())

        # Restore variables from training checkpoint
        checkpoint = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        checkpoint_path = checkpoint.model_checkpoint_path

        if FLAGS.remove_export:
            if os.path.isdir(FLAGS.export_dir):
                log_info('Removing old export')
                shutil.rmtree(FLAGS.export_dir)
        try:
            output_graph_path = os.path.join(FLAGS.export_dir, 'output_graph.pb')

            if not os.path.isdir(FLAGS.export_dir):
                os.makedirs(FLAGS.export_dir)

            # Freeze graph
            freeze_graph.freeze_graph_with_def_protos(
                input_graph_def=session.graph_def,
                input_saver_def=saver.as_saver_def(),
                input_checkpoint=checkpoint_path,
                output_node_names=','.join(node.op.name for node in six.itervalues(outputs)),
                restore_op_name=None,
                filename_tensor_name=None,
                output_graph=output_graph_path,
                clear_devices=False,
                initializer_nodes='')

            log_info('Models exported at %s' % (FLAGS.export_dir))
        except RuntimeError as e:
            log_error(str(e))


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

    if FLAGS.train or FLAGS.test:
        if len(FLAGS.worker_hosts) == 0:
            # Only one local task: this process (default case - no cluster)
            train()
            log_debug('Done.')
        else:
            # Create and start a server for the local task.
            server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
            if FLAGS.job_name == 'ps':
                # We are a parameter server and therefore we just wait for all workers to finish
                # by waiting for their stop tokens.
                with tf.Session(server.target) as session:
                    for worker in FLAGS.worker_hosts:
                        log_debug('Waiting for stop token...')
                        token = session.run(done_dequeues[FLAGS.task_index])
                        if token < 0:
                            log_debug('Got a kill switch token from worker %i.' % abs(token + 1))
                            break
                        log_debug('Got a stop token from worker %i.' % token)
                log_debug('Session closed.')
            elif FLAGS.job_name == 'worker':
                # We are a worker and therefore we have to do some work.
                # Assigns ops to the local worker by default.
                with tf.device(tf.train.replica_device_setter(
                               worker_device=worker_device,
                               cluster=cluster)):

                    # Do the training
                    train(server)

            log_debug('Server stopped.')

    # Are we the main process?
    if is_chief:
        # Doing solo/post-processing work just on the main process...
        # Exporting the model
        if FLAGS.export_dir:
            export()

    if len(FLAGS.one_shot_infer):
        do_single_file_inference(FLAGS.one_shot_infer)

    # Stopping the coordinator
    COORD.stop()

if __name__ == '__main__' :
    tf.app.run()
