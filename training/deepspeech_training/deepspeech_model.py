#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys

LOG_LEVEL_INDEX = sys.argv.index('--log_level') + 1 if '--log_level' in sys.argv else 0
DESIRED_LOG_LEVEL = sys.argv[LOG_LEVEL_INDEX] if 0 < LOG_LEVEL_INDEX < len(sys.argv) else '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = DESIRED_LOG_LEVEL

import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tfv1

tfv1.logging.set_verbosity({
    '0': tfv1.logging.DEBUG,
    '1': tfv1.logging.INFO,
    '2': tfv1.logging.WARN,
    '3': tfv1.logging.ERROR
}.get(DESIRED_LOG_LEVEL))

from .util.config import Config
from .util.feeding import audio_to_features
from .util.flags import FLAGS


def variable_on_cpu(name, shape, initializer):
    r"""
    Next we concern ourselves with graph creation.
    However, before we do so we must introduce a utility function ``variable_on_cpu()``
    used to create a variable in CPU memory.
    """
    # Use the /cpu:0 device for scoped operations
    with tf.device(Config.cpu_device):
        # Create or get apropos variable
        var = tfv1.get_variable(name=name, shape=shape, initializer=initializer)
    return var


def create_overlapping_windows(batch_x):
    batch_size = tf.shape(input=batch_x)[0]
    window_width = 2 * Config.n_context + 1
    num_channels = Config.n_input

    # Create a constant convolution filter using an identity matrix, so that the
    # convolution returns patches of the input tensor as is, and we can create
    # overlapping windows over the MFCCs.
    eye_filter = tf.constant(np.eye(window_width * num_channels)
                               .reshape(window_width, num_channels, window_width * num_channels), tf.float32)

    # Create overlapping windows
    batch_x = tf.nn.conv1d(input=batch_x, filters=eye_filter, stride=1, padding='SAME')

    # Remove dummy depth dimension and reshape into [batch_size, n_windows, window_width, n_input]
    batch_x = tf.reshape(batch_x, [batch_size, -1, window_width, num_channels])

    return batch_x


def dense(name, x, units, dropout_rate=None, relu=True, layer_norm=False):
    with tfv1.variable_scope(name):
        bias = variable_on_cpu('bias', [units], tfv1.zeros_initializer())
        weights = variable_on_cpu('weights', [x.shape[-1], units], tfv1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"))

    output = tf.nn.bias_add(tf.matmul(x, weights), bias)

    if relu:
        output = tf.minimum(tf.nn.relu(output), FLAGS.relu_clip)

    if layer_norm:
        with tfv1.variable_scope(name):
            output = tf.contrib.layers.layer_norm(output)

    if dropout_rate is not None:
        output = tf.nn.dropout(output, rate=dropout_rate)

    return output


def rnn_impl_lstmblockfusedcell(x, seq_length, previous_state, reuse):
    with tfv1.variable_scope('cudnn_lstm/rnn/multi_rnn_cell/cell_0'):
        fw_cell = tf.contrib.rnn.LSTMBlockFusedCell(Config.n_cell_dim,
                                                    forget_bias=0,
                                                    reuse=reuse,
                                                    name='cudnn_compatible_lstm_cell')

        output, output_state = fw_cell(inputs=x,
                                       dtype=tf.float32,
                                       sequence_length=seq_length,
                                       initial_state=previous_state)

    return output, output_state


def rnn_impl_cudnn_rnn(x, seq_length, previous_state, _):
    assert previous_state is None # 'Passing previous state not supported with CuDNN backend'

    # Hack: CudnnLSTM works similarly to Keras layers in that when you instantiate
    # the object it creates the variables, and then you just call it several times
    # to enable variable re-use. Because all of our code is structure in an old
    # school TensorFlow structure where you can just call tf.get_variable again with
    # reuse=True to reuse variables, we can't easily make use of the object oriented
    # way CudnnLSTM is implemented, so we save a singleton instance in the function,
    # emulating a static function variable.
    if not rnn_impl_cudnn_rnn.cell:
        # Forward direction cell:
        fw_cell = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers=1,
                                                 num_units=Config.n_cell_dim,
                                                 input_mode='linear_input',
                                                 direction='unidirectional',
                                                 dtype=tf.float32)
        rnn_impl_cudnn_rnn.cell = fw_cell

    output, output_state = rnn_impl_cudnn_rnn.cell(inputs=x,
                                                   sequence_lengths=seq_length)

    return output, output_state

rnn_impl_cudnn_rnn.cell = None


def rnn_impl_static_rnn(x, seq_length, previous_state, reuse):
    with tfv1.variable_scope('cudnn_lstm/rnn/multi_rnn_cell'):
        # Forward direction cell:
        fw_cell = tfv1.nn.rnn_cell.LSTMCell(Config.n_cell_dim,
                                            forget_bias=0,
                                            reuse=reuse,
                                            name='cudnn_compatible_lstm_cell')

        # Split rank N tensor into list of rank N-1 tensors
        x = [x[l] for l in range(x.shape[0])]

        output, output_state = tfv1.nn.static_rnn(cell=fw_cell,
                                                  inputs=x,
                                                  sequence_length=seq_length,
                                                  initial_state=previous_state,
                                                  dtype=tf.float32,
                                                  scope='cell_0')

        output = tf.concat(output, 0)

    return output, output_state


def create_model(batch_x, seq_length, dropout, reuse=False, batch_size=None, previous_state=None, overlap=True, rnn_impl=rnn_impl_lstmblockfusedcell):
    layers = {}

    # Input shape: [batch_size, n_steps, n_input + 2*n_input*n_context]
    if not batch_size:
        batch_size = tf.shape(input=batch_x)[0]

    # Create overlapping feature windows if needed
    if overlap:
        batch_x = create_overlapping_windows(batch_x)

    # Reshaping `batch_x` to a tensor with shape `[n_steps*batch_size, n_input + 2*n_input*n_context]`.
    # This is done to prepare the batch for input into the first layer which expects a tensor of rank `2`.

    # Permute n_steps and batch_size
    batch_x = tf.transpose(a=batch_x, perm=[1, 0, 2, 3])
    # Reshape to prepare input for first layer
    batch_x = tf.reshape(batch_x, [-1, Config.n_input + 2*Config.n_input*Config.n_context]) # (n_steps*batch_size, n_input + 2*n_input*n_context)
    layers['input_reshaped'] = batch_x

    # The next three blocks will pass `batch_x` through three hidden layers with
    # clipped RELU activation and dropout.
    layers['layer_1'] = layer_1 = dense('layer_1', batch_x, Config.n_hidden_1, dropout_rate=dropout[0], layer_norm=FLAGS.layer_norm)
    layers['layer_2'] = layer_2 = dense('layer_2', layer_1, Config.n_hidden_2, dropout_rate=dropout[1], layer_norm=FLAGS.layer_norm)
    layers['layer_3'] = layer_3 = dense('layer_3', layer_2, Config.n_hidden_3, dropout_rate=dropout[2], layer_norm=FLAGS.layer_norm)

    # `layer_3` is now reshaped into `[n_steps, batch_size, 2*n_cell_dim]`,
    # as the LSTM RNN expects its input to be of shape `[max_time, batch_size, input_size]`.
    layer_3 = tf.reshape(layer_3, [-1, batch_size, Config.n_hidden_3])

    # Run through parametrized RNN implementation, as we use different RNNs
    # for training and inference
    output, output_state = rnn_impl(layer_3, seq_length, previous_state, reuse)

    # Reshape output from a tensor of shape [n_steps, batch_size, n_cell_dim]
    # to a tensor of shape [n_steps*batch_size, n_cell_dim]
    output = tf.reshape(output, [-1, Config.n_cell_dim])
    layers['rnn_output'] = output
    layers['rnn_output_state'] = output_state

    # Now we feed `output` to the fifth hidden layer with clipped RELU activation
    layers['layer_5'] = layer_5 = dense('layer_5', output, Config.n_hidden_5, dropout_rate=dropout[5], layer_norm=FLAGS.layer_norm)

    # Now we apply a final linear layer creating `n_classes` dimensional vectors, the logits.
    layers['layer_6'] = layer_6 = dense('layer_6', layer_5, Config.n_hidden_6, relu=False)

    # Finally we reshape layer_6 from a tensor of shape [n_steps*batch_size, n_hidden_6]
    # to the slightly more useful shape [n_steps, batch_size, n_hidden_6].
    # Note, that this differs from the input in that it is time-major.
    layer_6 = tf.reshape(layer_6, [-1, batch_size, Config.n_hidden_6], name='raw_logits')
    layers['raw_logits'] = layer_6

    # Output shape: [n_steps, batch_size, n_hidden_6]
    return layer_6, layers


def create_inference_graph(batch_size=1, n_steps=16, tflite=False):
    batch_size = batch_size if batch_size > 0 else None

    # Create feature computation graph
    input_samples = tfv1.placeholder(tf.float32, [Config.audio_window_samples], 'input_samples')
    samples = tf.expand_dims(input_samples, -1)
    mfccs, _ = audio_to_features(samples, FLAGS.audio_sample_rate)
    mfccs = tf.identity(mfccs, name='mfccs')

    # Input tensor will be of shape [batch_size, n_steps, 2*n_context+1, n_input]
    # This shape is read by the native_client in DS_CreateModel to know the
    # value of n_steps, n_context and n_input. Make sure you update the code
    # there if this shape is changed.
    input_tensor = tfv1.placeholder(tf.float32, [batch_size, n_steps if n_steps > 0 else None, 2 * Config.n_context + 1, Config.n_input], name='input_node')
    seq_length = tfv1.placeholder(tf.int32, [batch_size], name='input_lengths')

    if batch_size <= 0:
        # no state management since n_step is expected to be dynamic too (see below)
        previous_state = None
    else:
        previous_state_c = tfv1.placeholder(tf.float32, [batch_size, Config.n_cell_dim], name='previous_state_c')
        previous_state_h = tfv1.placeholder(tf.float32, [batch_size, Config.n_cell_dim], name='previous_state_h')

        previous_state = tf.nn.rnn_cell.LSTMStateTuple(previous_state_c, previous_state_h)

    # One rate per layer
    no_dropout = [None] * 6

    if tflite:
        rnn_impl = rnn_impl_static_rnn
    else:
        rnn_impl = rnn_impl_lstmblockfusedcell

    logits, layers = create_model(batch_x=input_tensor,
                                  batch_size=batch_size,
                                  seq_length=seq_length if not FLAGS.export_tflite else None,
                                  dropout=no_dropout,
                                  previous_state=previous_state,
                                  overlap=False,
                                  rnn_impl=rnn_impl)

    # TF Lite runtime will check that input dimensions are 1, 2 or 4
    # by default we get 3, the middle one being batch_size which is forced to
    # one on inference graph, so remove that dimension
    if tflite:
        logits = tf.squeeze(logits, [1])

    # Apply softmax for CTC decoder
    probs = tf.nn.softmax(logits, name='logits')

    if batch_size <= 0:
        if tflite:
            raise NotImplementedError('dynamic batch_size does not support tflite nor streaming')
        if n_steps > 0:
            raise NotImplementedError('dynamic batch_size expect n_steps to be dynamic too')
        return (
            {
                'input': input_tensor,
                'input_lengths': seq_length,
            },
            {
                'outputs': probs,
            },
            layers
        )

    new_state_c, new_state_h = layers['rnn_output_state']
    new_state_c = tf.identity(new_state_c, name='new_state_c')
    new_state_h = tf.identity(new_state_h, name='new_state_h')

    inputs = {
        'input': input_tensor,
        'previous_state_c': previous_state_c,
        'previous_state_h': previous_state_h,
        'input_samples': input_samples,
    }

    if not FLAGS.export_tflite:
        inputs['input_lengths'] = seq_length

    outputs = {
        'outputs': probs,
        'new_state_c': new_state_c,
        'new_state_h': new_state_h,
        'mfccs': mfccs,
    }

    return inputs, outputs, layers
