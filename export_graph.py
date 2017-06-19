#!/usr/bin/env python

# export the step graphs of TensorFlow into a final binary graph for use with the Native Client
# ./export_graph.py --checkpoint_dir data/ckpt --export_dir data/graph

# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import os
import sys

log_level_index = sys.argv.index('--log_level') + 1 if '--log_level' in sys.argv else 0
os.environ['TF_CPP_MIN_LOG_LEVEL'] = sys.argv[log_level_index] if log_level_index > 0 and log_level_index < len(sys.argv) else '3'

import shutil
import tensorflow as tf
import inspect

from six.moves import filter
from tensorflow.contrib.session_bundle import exporter
from tensorflow.python.tools import freeze_graph
from util.gpu import get_available_gpus
from xdg import BaseDirectory as xdg


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
tf.app.flags.DEFINE_string  ('coord_retries',    100,         'number of tries of workers connecting to training coordinator before failing')
tf.app.flags.DEFINE_string  ('coord_host',       'localhost', 'coordination server host')
tf.app.flags.DEFINE_integer ('coord_port',       2500,        'coordination server port')
tf.app.flags.DEFINE_integer ('iters_per_worker', 1,           'number of train or inference iterations per worker before results are sent back to coordinator')

# Global Constants
# ================

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

# Exporting

tf.app.flags.DEFINE_string  ('export_dir',       '',          'directory in which exported models are stored - if omitted, the model won\'t get exported')
tf.app.flags.DEFINE_integer ('export_version',   1,           'version number of the exported model')
tf.app.flags.DEFINE_boolean ('remove_export',    False,       'wether to remove old exported models')

# Reporting

tf.app.flags.DEFINE_integer ('log_level',        1,           'log level for console logs - 0: INFO, 1: WARN, 2: ERROR, 3: FATAL')
tf.app.flags.DEFINE_boolean ('log_traffic',      False,       'log cluster transaction and traffic information during debug logging')

tf.app.flags.DEFINE_string  ('wer_log_pattern',  '',          'pattern for machine readable global logging of WER progress; has to contain %%s, %%s and %%f for the set name, the date and the float respectively; example: "GLOBAL LOG: logwer(\'12ade231\', %%s, %%s, %%f)" would result in some entry like "GLOBAL LOG: logwer(\'12ade231\', \'train\', \'2017-05-18T03:09:48-0700\', 0.05)"; if omitted (default), there will be no logging')

tf.app.flags.DEFINE_boolean ('log_placement',    False,       'wether to log device placement of the operators to the console')
tf.app.flags.DEFINE_integer ('report_count',     10,          'number of phrases with lowest WER (best matching) to print out during a WER report')

tf.app.flags.DEFINE_string  ('summary_dir',      '',          'target directory for TensorBoard summaries - defaults to directory "deepspeech/summaries" within user\'s data home specified by the XDG Base Directory Specification')
tf.app.flags.DEFINE_integer ('summary_secs',     0,           'interval in seconds for saving TensorBoard summaries - if 0, no summaries will be written')

# Geometry

tf.app.flags.DEFINE_integer ('n_hidden',         2048,        'layer width to use when initialising layers')

# Initialization

tf.app.flags.DEFINE_integer ('random_seed',      4567,        'default random seed that is used to initialize variables')
tf.app.flags.DEFINE_float   ('default_stddev',   0.046875,    'default standard deviation to use when initialising weights and biases')

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
    session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=FLAGS.log_placement)

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
    n_character = 29 # TODO: Determine if this should be extended with other punctuation

    # The number of units in the sixth layer
    global n_hidden_6
    n_hidden_6 = n_character

    # Assign default values for standard deviation
    for var in ['b1', 'h1', 'b2', 'h2', 'b3', 'h3', 'b5', 'h5', 'b6', 'h6']:
        val = getattr(FLAGS, '%s_stddev' % var)
        if val is None:
            setattr(FLAGS, '%s_stddev' % var, FLAGS.default_stddev)

    # Queues that are used to gracefully stop parameter servers.
    # Each queue stands for one ps. A finishing worker sends a token to each queue befor joining/quitting.
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


# Logging functions
# =================

def prefix_print(prefix, message):
    print(prefix + ('\n' + prefix).join(message.split('\n')))

def log_debug(message):
    if FLAGS.log_level == 0:
        prefix_print('D ', str(message))

def log_traffic(message):
    if FLAGS.log_traffic:
        log_debug(message)

def log_info(message):
    if FLAGS.log_level <= 1:
        prefix_print('I ', str(message))

def log_warn(message):
    if FLAGS.log_level <= 2:
        prefix_print('W ', str(message))

def log_error(message):
    if FLAGS.log_level <= 3:
        prefix_print('E ', str(message))


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
    h1 = variable_on_worker_level('h1', [n_input + 2*n_input*n_context, n_hidden_1], tf.random_normal_initializer(stddev=FLAGS.h1_stddev))
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
    b5 = variable_on_worker_level('b5', [n_hidden_5], tf.random_normal_initializer(stddev=FLAGS.b5_stddev))
    h5 = variable_on_worker_level('h5', [(2 * n_cell_dim), n_hidden_5], tf.random_normal_initializer(stddev=FLAGS.h5_stddev))
    layer_5 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(outputs, h5), b5)), FLAGS.relu_clip)
    layer_5 = tf.nn.dropout(layer_5, (1.0 - dropout[5]))

    # Now we apply the weight matrix `h6` and bias `b6` to the output of `layer_5`
    # creating `n_classes` dimensional vectors, the logits.
    b6 = variable_on_worker_level('b6', [n_hidden_6], tf.random_normal_initializer(stddev=FLAGS.b6_stddev))
    h6 = variable_on_worker_level('h6', [n_hidden_5, n_hidden_6], tf.random_normal_initializer(stddev=FLAGS.h6_stddev))
    layer_6 = tf.add(tf.matmul(layer_5, h6), b6)

    # Finally we reshape layer_6 from a tensor of shape [n_steps*batch_size, n_hidden_6]
    # to the slightly more useful shape [n_steps, batch_size, n_hidden_6].
    # Note, that this differs from the input in that it is time-major.
    layer_6 = tf.reshape(layer_6, [-1, batch_x_shape[0], n_hidden_6])

    # Output shape: [n_steps, batch_size, n_hidden_6]
    return layer_6


# Execution
# =========

def export():
    r'''
    Restores the trained variables into a simpler graph that will be exported for serving.
    '''
    log_info('Exporting the model...')
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
        log_info('Restored checkpoint at training epoch %d' % (int(checkpoint_path.split('-')[-1]) + 1))

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
                log_info('Removing old export')
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

            log_info('Models exported at %s' % (FLAGS.export_dir))
        except RuntimeError:
            log_error(sys.exc_info()[1])


def main(_) :
    print("init globals")
    initialize_globals()
    print("export")
    export()


if __name__ == '__main__' :
    tf.app.run()
