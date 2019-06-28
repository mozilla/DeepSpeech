#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import os
import sys

LOG_LEVEL_INDEX = sys.argv.index('--log_level') + 1 if '--log_level' in sys.argv else 0
os.environ['TF_CPP_MIN_LOG_LEVEL'] = sys.argv[LOG_LEVEL_INDEX] if 0 < LOG_LEVEL_INDEX < len(sys.argv) else '3'

import time
import numpy as np
import progressbar
import shutil
import tensorflow as tf
import tensorflow.compat.v1 as tfv1

from datetime import datetime
from ds_ctcdecoder import ctc_beam_search_decoder, Scorer
from evaluate import evaluate
from six.moves import zip, range
from tensorflow.python.tools import freeze_graph, strip_unused_lib
from util.config import Config, initialize_globals
from util.feeding import create_dataset, samples_to_mfccs, audiofile_to_features
from util.flags import create_flags, FLAGS
from util.logging import log_info, log_error, log_debug, log_progress, create_progressbar


# Graph Creation
# ==============

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
    batch_size = tf.shape(batch_x)[0]
    window_width = 2 * Config.n_context + 1
    num_channels = Config.n_input

    # Create a constant convolution filter using an identity matrix, so that the
    # convolution returns patches of the input tensor as is, and we can create
    # overlapping windows over the MFCCs.
    eye_filter = tf.constant(np.eye(window_width * num_channels)
                               .reshape(window_width, num_channels, window_width * num_channels), tf.float32) # pylint: disable=bad-continuation

    # Create overlapping windows
    batch_x = tf.nn.conv1d(batch_x, eye_filter, stride=1, padding='SAME')

    # Remove dummy depth dimension and reshape into [batch_size, n_windows, window_width, n_input]
    batch_x = tf.reshape(batch_x, [batch_size, -1, window_width, num_channels])

    return batch_x


def dense(name, x, units, dropout_rate=None, relu=True):
    with tfv1.variable_scope(name):
        bias = variable_on_cpu('bias', [units], tf.zeros_initializer())
        weights = variable_on_cpu('weights', [x.shape[-1], units], tf.contrib.layers.xavier_initializer())

    output = tf.nn.bias_add(tf.matmul(x, weights), bias)

    if relu:
        output = tf.minimum(tf.nn.relu(output), FLAGS.relu_clip)

    if dropout_rate is not None:
        output = tf.nn.dropout(output, rate=dropout_rate)

    return output


def rnn_impl_lstmblockfusedcell(x, seq_length, previous_state, reuse):
    # Forward direction cell:
    fw_cell = tf.contrib.rnn.LSTMBlockFusedCell(Config.n_cell_dim, reuse=reuse)

    output, output_state = fw_cell(inputs=x,
                                   dtype=tf.float32,
                                   sequence_length=seq_length,
                                   initial_state=previous_state)

    return output, output_state


def rnn_impl_static_rnn(x, seq_length, previous_state, reuse):
    # Forward direction cell:
    fw_cell = tf.nn.rnn_cell.LSTMCell(Config.n_cell_dim, reuse=reuse)

    # Split rank N tensor into list of rank N-1 tensors
    x = [x[l] for l in range(x.shape[0])]

    # We parametrize the RNN implementation as the training and inference graph
    # need to do different things here.
    output, output_state = tf.nn.static_rnn(cell=fw_cell,
                                            inputs=x,
                                            initial_state=previous_state,
                                            dtype=tf.float32,
                                            sequence_length=seq_length)
    output = tf.concat(output, 0)

    return output, output_state


def create_model(batch_x, seq_length, dropout, reuse=False, previous_state=None, overlap=True, rnn_impl=rnn_impl_lstmblockfusedcell):
    layers = {}

    # Input shape: [batch_size, n_steps, n_input + 2*n_input*n_context]
    batch_size = tf.shape(batch_x)[0]

    # Create overlapping feature windows if needed
    if overlap:
        batch_x = create_overlapping_windows(batch_x)

    # Reshaping `batch_x` to a tensor with shape `[n_steps*batch_size, n_input + 2*n_input*n_context]`.
    # This is done to prepare the batch for input into the first layer which expects a tensor of rank `2`.

    # Permute n_steps and batch_size
    batch_x = tf.transpose(batch_x, [1, 0, 2, 3])
    # Reshape to prepare input for first layer
    batch_x = tf.reshape(batch_x, [-1, Config.n_input + 2*Config.n_input*Config.n_context]) # (n_steps*batch_size, n_input + 2*n_input*n_context)
    layers['input_reshaped'] = batch_x

    # The next three blocks will pass `batch_x` through three hidden layers with
    # clipped RELU activation and dropout.
    layers['layer_1'] = layer_1 = dense('layer_1', batch_x, Config.n_hidden_1, dropout_rate=dropout[0])
    layers['layer_2'] = layer_2 = dense('layer_2', layer_1, Config.n_hidden_2, dropout_rate=dropout[1])
    layers['layer_3'] = layer_3 = dense('layer_3', layer_2, Config.n_hidden_3, dropout_rate=dropout[2])

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
    layers['layer_5'] = layer_5 = dense('layer_5', output, Config.n_hidden_5, dropout_rate=dropout[5])

    # Now we apply a final linear layer creating `n_classes` dimensional vectors, the logits.
    layers['layer_6'] = layer_6 = dense('layer_6', layer_5, Config.n_hidden_6, relu=False)

    # Finally we reshape layer_6 from a tensor of shape [n_steps*batch_size, n_hidden_6]
    # to the slightly more useful shape [n_steps, batch_size, n_hidden_6].
    # Note, that this differs from the input in that it is time-major.
    layer_6 = tf.reshape(layer_6, [-1, batch_size, Config.n_hidden_6], name='raw_logits')
    layers['raw_logits'] = layer_6

    # Output shape: [n_steps, batch_size, n_hidden_6]
    return layer_6, layers


# Accuracy and Loss
# =================

# In accord with 'Deep Speech: Scaling up end-to-end speech recognition'
# (http://arxiv.org/abs/1412.5567),
# the loss function used by our network should be the CTC loss function
# (http://www.cs.toronto.edu/~graves/preprint.pdf).
# Conveniently, this loss function is implemented in TensorFlow.
# Thus, we can simply make use of this implementation to define our loss.

def calculate_mean_edit_distance_and_loss(iterator, dropout, reuse):
    r'''
    This routine beam search decodes a mini-batch and calculates the loss and mean edit distance.
    Next to total and average loss it returns the mean edit distance,
    the decoded result and the batch's original Y.
    '''
    # Obtain the next batch of data
    (batch_x, batch_seq_len), batch_y = iterator.get_next()

    # Calculate the logits of the batch
    logits, _ = create_model(batch_x, batch_seq_len, dropout, reuse=reuse)

    # Compute the CTC loss using TensorFlow's `ctc_loss`
    total_loss = tfv1.nn.ctc_loss(labels=batch_y, inputs=logits, sequence_length=batch_seq_len)

    # Calculate the average loss across the batch
    avg_loss = tf.reduce_mean(total_loss)

    # Finally we return the average loss
    return avg_loss


# Adam Optimization
# =================

# In contrast to 'Deep Speech: Scaling up end-to-end speech recognition'
# (http://arxiv.org/abs/1412.5567),
# in which 'Nesterov's Accelerated Gradient Descent'
# (www.cs.toronto.edu/~fritz/absps/momentum.pdf) was used,
# we will use the Adam method for optimization (http://arxiv.org/abs/1412.6980),
# because, generally, it requires less fine-tuning.
def create_optimizer():
    optimizer = tfv1.train.AdamOptimizer(learning_rate=FLAGS.learning_rate,
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

def get_tower_results(iterator, optimizer, dropout_rates):
    r'''
    With this preliminary step out of the way, we can for each GPU introduce a
    tower for which's batch we calculate and return the optimization gradients
    and the average loss across towers.
    '''
    # To calculate the mean of the losses
    tower_avg_losses = []

    # Tower gradients to return
    tower_gradients = []

    with tfv1.variable_scope(tfv1.get_variable_scope()):
        # Loop over available_devices
        for i in range(len(Config.available_devices)):
            # Execute operations of tower i on device i
            device = Config.available_devices[i]
            with tf.device(device):
                # Create a scope for all operations of tower i
                with tf.name_scope('tower_%d' % i):
                    # Calculate the avg_loss and mean_edit_distance and retrieve the decoded
                    # batch along with the original batch's labels (Y) of this tower
                    avg_loss = calculate_mean_edit_distance_and_loss(iterator, dropout_rates, reuse=i > 0)

                    # Allow for variables to be re-used by the next tower
                    tfv1.get_variable_scope().reuse_variables()

                    # Retain tower's avg losses
                    tower_avg_losses.append(avg_loss)

                    # Compute gradients for model parameters using tower's mini-batch
                    gradients = optimizer.compute_gradients(avg_loss)

                    # Retain tower's gradients
                    tower_gradients.append(gradients)


    avg_loss_across_towers = tf.reduce_mean(tower_avg_losses, 0)

    tfv1.summary.scalar(name='step_loss', tensor=avg_loss_across_towers, collections=['step_summaries'])

    # Return gradients and the average loss
    return tower_gradients, avg_loss_across_towers


def average_gradients(tower_gradients):
    r'''
    A routine for computing each variable's average of the gradients obtained from the GPUs.
    Note also that this code acts as a synchronization point as it requires all
    GPUs to be finished with their mini-batch before it can run to completion.
    '''
    # List of average gradients to return to the caller
    average_grads = []

    # Run this on cpu_device to conserve GPU memory
    with tf.device(Config.cpu_device):
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
    name = variable.name.replace(':', '_')
    mean = tf.reduce_mean(variable)
    tfv1.summary.scalar(name='%s/mean'   % name, tensor=mean)
    tfv1.summary.scalar(name='%s/sttdev' % name, tensor=tf.sqrt(tf.reduce_mean(tf.square(variable - mean))))
    tfv1.summary.scalar(name='%s/max'    % name, tensor=tf.reduce_max(variable))
    tfv1.summary.scalar(name='%s/min'    % name, tensor=tf.reduce_min(variable))
    tfv1.summary.histogram(name=name, values=variable)
    if gradient is not None:
        if isinstance(gradient, tf.IndexedSlices):
            grad_values = gradient.values
        else:
            grad_values = gradient
        if grad_values is not None:
            tfv1.summary.histogram(name='%s/gradients' % name, values=grad_values)


def log_grads_and_vars(grads_and_vars):
    r'''
    Let's also introduce a helper function for logging collections of gradient/variable tuples.
    '''
    for gradient, variable in grads_and_vars:
        log_variable(variable, gradient=gradient)


def try_loading(session, saver, checkpoint_filename, caption):
    try:
        checkpoint = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir, checkpoint_filename)
        if not checkpoint:
            return False
        checkpoint_path = checkpoint.model_checkpoint_path
        saver.restore(session, checkpoint_path)
        restored_step = session.run(tfv1.train.get_global_step())
        log_info('Restored variables from %s checkpoint at %s, step %d' % (caption, checkpoint_path, restored_step))
        return True
    except tf.errors.InvalidArgumentError as e:
        log_error(str(e))
        log_error('The checkpoint in {0} does not match the shapes of the model.'
                  ' Did you change alphabet.txt or the --n_hidden parameter'
                  ' between train runs using the same checkpoint dir? Try moving'
                  ' or removing the contents of {0}.'.format(checkpoint_path))
        sys.exit(1)


def train():
    # Create training and validation datasets
    train_set = create_dataset(FLAGS.train_files.split(','),
                               batch_size=FLAGS.train_batch_size,
                               cache_path=FLAGS.feature_cache)

    iterator = tfv1.data.Iterator.from_structure(tfv1.data.get_output_types(train_set),
                                                 tfv1.data.get_output_shapes(train_set),
                                                 output_classes=tfv1.data.get_output_classes(train_set))

    # Make initialization ops for switching between the two sets
    train_init_op = iterator.make_initializer(train_set)

    if FLAGS.dev_files:
        dev_csvs = FLAGS.dev_files.split(',')
        dev_sets = [create_dataset([csv], batch_size=FLAGS.dev_batch_size) for csv in dev_csvs]
        dev_init_ops = [iterator.make_initializer(dev_set) for dev_set in dev_sets]

    # Dropout
    dropout_rates = [tfv1.placeholder(tf.float32, name='dropout_{}'.format(i)) for i in range(6)]
    dropout_feed_dict = {
        dropout_rates[0]: FLAGS.dropout_rate,
        dropout_rates[1]: FLAGS.dropout_rate2,
        dropout_rates[2]: FLAGS.dropout_rate3,
        dropout_rates[3]: FLAGS.dropout_rate4,
        dropout_rates[4]: FLAGS.dropout_rate5,
        dropout_rates[5]: FLAGS.dropout_rate6,
    }
    no_dropout_feed_dict = {
        rate: 0. for rate in dropout_rates
    }

    # Building the graph
    optimizer = create_optimizer()
    gradients, loss = get_tower_results(iterator, optimizer, dropout_rates)

    # Average tower gradients across GPUs
    avg_tower_gradients = average_gradients(gradients)
    log_grads_and_vars(avg_tower_gradients)

    # global_step is automagically incremented by the optimizer
    global_step = tfv1.train.get_or_create_global_step()
    apply_gradient_op = optimizer.apply_gradients(avg_tower_gradients, global_step=global_step)

    # Summaries
    step_summaries_op = tfv1.summary.merge_all('step_summaries')
    step_summary_writers = {
        'train': tfv1.summary.FileWriter(os.path.join(FLAGS.summary_dir, 'train'), max_queue=120),
        'dev': tfv1.summary.FileWriter(os.path.join(FLAGS.summary_dir, 'dev'), max_queue=120)
    }

    # Checkpointing
    checkpoint_saver = tfv1.train.Saver(max_to_keep=FLAGS.max_to_keep)
    checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'train')
    checkpoint_filename = 'checkpoint'

    best_dev_saver = tfv1.train.Saver(max_to_keep=1)
    best_dev_path = os.path.join(FLAGS.checkpoint_dir, 'best_dev')
    best_dev_filename = 'best_dev_checkpoint'

    initializer = tfv1.global_variables_initializer()

    with tfv1.Session() as session:
        log_debug('Session opened.')

        tfv1.get_default_graph().finalize()

        # Loading or initializing
        loaded = False
        if FLAGS.load in ['auto', 'last']:
            loaded = try_loading(session, checkpoint_saver, checkpoint_filename, 'most recent')
        if not loaded and FLAGS.load in ['auto', 'best']:
            loaded = try_loading(session, best_dev_saver, best_dev_filename, 'best validation')
        if not loaded:
            if FLAGS.load in ['auto', 'init']:
                log_info('Initializing variables...')
                session.run(initializer)
            else:
                log_error('Unable to load %s model from specified checkpoint dir'
                          ' - consider using load option "auto" or "init".' % FLAGS.load)
                sys.exit(1)

        def run_set(set_name, epoch, init_op, dataset=None):
            is_train = set_name == 'train'
            train_op = apply_gradient_op if is_train else []
            feed_dict = dropout_feed_dict if is_train else no_dropout_feed_dict

            total_loss = 0.0
            step_count = 0

            step_summary_writer = step_summary_writers.get(set_name)
            checkpoint_time = time.time()

            # Setup progress bar
            class LossWidget(progressbar.widgets.FormatLabel):
                def __init__(self):
                    progressbar.widgets.FormatLabel.__init__(self, format='Loss: %(mean_loss)f')

                def __call__(self, progress, data, **kwargs):
                    data['mean_loss'] = total_loss / step_count if step_count else 0.0
                    return progressbar.widgets.FormatLabel.__call__(self, progress, data, **kwargs)

            prefix = 'Epoch {} | {:>10}'.format(epoch, 'Training' if is_train else 'Validation')
            widgets = [' | ', progressbar.widgets.Timer(),
                       ' | Steps: ', progressbar.widgets.Counter(),
                       ' | ', LossWidget()]
            suffix = ' | Dataset: {}'.format(dataset) if dataset else None
            pbar = create_progressbar(prefix=prefix, widgets=widgets, suffix=suffix).start()

            # Initialize iterator to the appropriate dataset
            session.run(init_op)

            # Batch loop
            while True:
                try:
                    _, current_step, batch_loss, step_summary = \
                        session.run([train_op, global_step, loss, step_summaries_op],
                                    feed_dict=feed_dict)
                except tf.errors.OutOfRangeError:
                    break

                total_loss += batch_loss
                step_count += 1

                pbar.update(step_count)

                step_summary_writer.add_summary(step_summary, current_step)

                if is_train and FLAGS.checkpoint_secs > 0 and time.time() - checkpoint_time > FLAGS.checkpoint_secs:
                    checkpoint_saver.save(session, checkpoint_path, global_step=current_step)
                    checkpoint_time = time.time()

            pbar.finish()
            mean_loss = total_loss / step_count if step_count > 0 else 0.0
            return mean_loss, step_count

        log_info('STARTING Optimization')
        train_start_time = datetime.utcnow()
        best_dev_loss = float('inf')
        dev_losses = []
        try:
            for epoch in range(FLAGS.epochs):
                # Training
                log_progress('Training epoch %d...' % epoch)
                train_loss, _ = run_set('train', epoch, train_init_op)
                log_progress('Finished training epoch %d - loss: %f' % (epoch, train_loss))
                checkpoint_saver.save(session, checkpoint_path, global_step=global_step)

                if FLAGS.dev_files:
                    # Validation
                    dev_loss = 0.0
                    total_steps = 0
                    for csv, init_op in zip(dev_csvs, dev_init_ops):
                        log_progress('Validating epoch %d on %s...' % (epoch, csv))
                        set_loss, steps = run_set('dev', epoch, init_op, dataset=csv)
                        dev_loss += set_loss * steps
                        total_steps += steps
                        log_progress('Finished validating epoch %d on %s - loss: %f' % (epoch, csv, set_loss))
                    dev_loss = dev_loss / total_steps

                    dev_losses.append(dev_loss)

                    if dev_loss < best_dev_loss:
                        best_dev_loss = dev_loss
                        save_path = best_dev_saver.save(session, best_dev_path, global_step=global_step, latest_filename=best_dev_filename)
                        log_info("Saved new best validating model with loss %f to: %s" % (best_dev_loss, save_path))

                    # Early stopping
                    if FLAGS.early_stop and len(dev_losses) >= FLAGS.es_steps:
                        mean_loss = np.mean(dev_losses[-FLAGS.es_steps:-1])
                        std_loss = np.std(dev_losses[-FLAGS.es_steps:-1])
                        dev_losses = dev_losses[-FLAGS.es_steps:]
                        log_debug('Checking for early stopping (last %d steps) validation loss: '
                                  '%f, with standard deviation: %f and mean: %f' %
                                  (FLAGS.es_steps, dev_losses[-1], std_loss, mean_loss))
                        if dev_losses[-1] > np.max(dev_losses[:-1]) or \
                           (abs(dev_losses[-1] - mean_loss) < FLAGS.es_mean_th and std_loss < FLAGS.es_std_th):
                            log_info('Early stop triggered as (for last %d steps) validation loss:'
                                     ' %f with standard deviation: %f and mean: %f' %
                                     (FLAGS.es_steps, dev_losses[-1], std_loss, mean_loss))
                            break
        except KeyboardInterrupt:
            pass
        log_info('FINISHED optimization in {}'.format(datetime.utcnow() - train_start_time))
    log_debug('Session closed.')


def test():
    evaluate(FLAGS.test_files.split(','), create_model, try_loading)


def create_inference_graph(batch_size=1, n_steps=16, tflite=False):
    batch_size = batch_size if batch_size > 0 else None

    # Create feature computation graph
    input_samples = tfv1.placeholder(tf.float32, [Config.audio_window_samples], 'input_samples')
    samples = tf.expand_dims(input_samples, -1)
    mfccs, _ = samples_to_mfccs(samples, FLAGS.audio_sample_rate)
    mfccs = tf.identity(mfccs, name='mfccs')

    # Input tensor will be of shape [batch_size, n_steps, 2*n_context+1, n_input]
    # This shape is read by the native_client in DS_CreateModel to know the
    # value of n_steps, n_context and n_input. Make sure you update the code
    # there if this shape is changed.
    input_tensor = tfv1.placeholder(tf.float32, [batch_size, n_steps if n_steps > 0 else None, 2 * Config.n_context + 1, Config.n_input], name='input_node')
    seq_length = tfv1.placeholder(tf.int32, [batch_size], name='input_lengths')

    if batch_size <= 0:
        # no state management since n_step is expected to be dynamic too (see below)
        previous_state = previous_state_c = previous_state_h = None
    else:
        previous_state_c = tfv1.placeholder(tf.float32, [batch_size, Config.n_cell_dim], name='previous_state_c')
        previous_state_h = tfv1.placeholder(tf.float32, [batch_size, Config.n_cell_dim], name='previous_state_h')

        previous_state = tf.contrib.rnn.LSTMStateTuple(previous_state_c, previous_state_h)

    # One rate per layer
    no_dropout = [None] * 6

    if tflite:
        rnn_impl = rnn_impl_static_rnn
    else:
        rnn_impl = rnn_impl_lstmblockfusedcell

    logits, layers = create_model(batch_x=input_tensor,
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
    logits = tf.nn.softmax(logits, name='logits')

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
                'outputs': logits,
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
        inputs.update({'input_lengths': seq_length})

    outputs = {
        'outputs': logits,
        'new_state_c': new_state_c,
        'new_state_h': new_state_h,
        'mfccs': mfccs,
    }

    return inputs, outputs, layers

def file_relative_read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


def export():
    r'''
    Restores the trained variables into a simpler graph that will be exported for serving.
    '''
    log_info('Exporting the model...')
    from tensorflow.python.framework.ops import Tensor, Operation

    inputs, outputs, _ = create_inference_graph(batch_size=FLAGS.export_batch_size, n_steps=FLAGS.n_steps, tflite=FLAGS.export_tflite)
    output_names_tensors = [tensor.op.name for tensor in outputs.values() if isinstance(tensor, Tensor)]
    output_names_ops = [op.name for op in outputs.values() if isinstance(op, Operation)]
    output_names = ",".join(output_names_tensors + output_names_ops)

    mapping = None
    if FLAGS.export_tflite:
        # Create a saver using variables from the above newly created graph
        # Training graph uses LSTMFusedCell, but the TFLite inference graph uses
        # a static RNN with a normal cell, so we need to rewrite the names to
        # match the training weights when restoring.
        def fixup(name):
            if name.startswith('rnn/lstm_cell/'):
                return name.replace('rnn/lstm_cell/', 'lstm_fused_cell/')
            return name

        mapping = {fixup(v.op.name): v for v in tf.global_variables()}

    saver = tfv1.train.Saver(mapping)

    # Restore variables from training checkpoint
    checkpoint = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    checkpoint_path = checkpoint.model_checkpoint_path

    output_filename = 'output_graph.pb'
    if FLAGS.remove_export:
        if os.path.isdir(FLAGS.export_dir):
            log_info('Removing old export')
            shutil.rmtree(FLAGS.export_dir)
    try:
        output_graph_path = os.path.join(FLAGS.export_dir, output_filename)

        if not os.path.isdir(FLAGS.export_dir):
            os.makedirs(FLAGS.export_dir)

        def do_graph_freeze(output_file=None, output_node_names=None, variables_blacklist=''):
            frozen = freeze_graph.freeze_graph_with_def_protos(
                input_graph_def=tfv1.get_default_graph().as_graph_def(),
                input_saver_def=saver.as_saver_def(),
                input_checkpoint=checkpoint_path,
                output_node_names=output_node_names,
                restore_op_name=None,
                filename_tensor_name=None,
                output_graph=output_file,
                clear_devices=False,
                variable_names_blacklist=variables_blacklist,
                initializer_nodes='')

            input_node_names = []
            return strip_unused_lib.strip_unused(
                input_graph_def=frozen,
                input_node_names=input_node_names,
                output_node_names=output_node_names.split(','),
                placeholder_type_enum=tf.float32.as_datatype_enum)

        if not FLAGS.export_tflite:
            frozen_graph = do_graph_freeze(output_node_names=output_names)
            frozen_graph.version = int(file_relative_read('GRAPH_VERSION').strip())

            # Add a no-op node to the graph with metadata information to be loaded by the native client
            metadata = frozen_graph.node.add()
            metadata.name = 'model_metadata'
            metadata.op = 'NoOp'
            metadata.attr['sample_rate'].i = FLAGS.audio_sample_rate
            metadata.attr['feature_win_len'].i = FLAGS.feature_win_len
            metadata.attr['feature_win_step'].i = FLAGS.feature_win_step
            if FLAGS.export_language:
                metadata.attr['language'].s = FLAGS.export_language.encode('ascii')

            with open(output_graph_path, 'wb') as fout:
                fout.write(frozen_graph.SerializeToString())
        else:
            frozen_graph = do_graph_freeze(output_node_names=output_names)
            output_tflite_path = os.path.join(FLAGS.export_dir, output_filename.replace('.pb', '.tflite'))

            converter = tf.lite.TFLiteConverter(frozen_graph, input_tensors=inputs.values(), output_tensors=outputs.values())
            converter.post_training_quantize = True
            # AudioSpectrogram and Mfcc ops are custom but have built-in kernels in TFLite
            converter.allow_custom_ops = True
            tflite_model = converter.convert()

            with open(output_tflite_path, 'wb') as fout:
                fout.write(tflite_model)

            log_info('Exported model for TF Lite engine as {}'.format(os.path.basename(output_tflite_path)))

        log_info('Models exported at %s' % (FLAGS.export_dir))
    except RuntimeError as e:
        log_error(str(e))


def do_single_file_inference(input_file_path):
    with tfv1.Session() as session:
        inputs, outputs, _ = create_inference_graph(batch_size=1, n_steps=-1)

        # Create a saver using variables from the above newly created graph
        saver = tfv1.train.Saver()

        # Restore variables from training checkpoint
        # TODO: This restores the most recent checkpoint, but if we use validation to counteract
        #       over-fitting, we may want to restore an earlier checkpoint.
        checkpoint = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if not checkpoint:
            log_error('Checkpoint directory ({}) does not contain a valid checkpoint state.'.format(FLAGS.checkpoint_dir))
            exit(1)

        checkpoint_path = checkpoint.model_checkpoint_path
        saver.restore(session, checkpoint_path)

        features, features_len = audiofile_to_features(input_file_path)
        previous_state_c = np.zeros([1, Config.n_cell_dim])
        previous_state_h = np.zeros([1, Config.n_cell_dim])

        # Add batch dimension
        features = tf.expand_dims(features, 0)
        features_len = tf.expand_dims(features_len, 0)

        # Evaluate
        features = create_overlapping_windows(features).eval(session=session)
        features_len = features_len.eval(session=session)

        logits = outputs['outputs'].eval(feed_dict={
            inputs['input']: features,
            inputs['input_lengths']: features_len,
            inputs['previous_state_c']: previous_state_c,
            inputs['previous_state_h']: previous_state_h,
        }, session=session)

        logits = np.squeeze(logits)

        scorer = Scorer(FLAGS.lm_alpha, FLAGS.lm_beta,
                        FLAGS.lm_binary_path, FLAGS.lm_trie_path,
                        Config.alphabet)
        decoded = ctc_beam_search_decoder(logits, Config.alphabet, FLAGS.beam_width, scorer=scorer)
        # Print highest probability result
        print(decoded[0][1])


def main(_):
    initialize_globals()

    if FLAGS.train_files:
        tfv1.reset_default_graph()
        tfv1.set_random_seed(FLAGS.random_seed)
        train()

    if FLAGS.test_files:
        tfv1.reset_default_graph()
        test()

    if FLAGS.export_dir:
        tfv1.reset_default_graph()
        export()

    if FLAGS.one_shot_infer:
        tfv1.reset_default_graph()
        do_single_file_inference(FLAGS.one_shot_infer)

if __name__ == '__main__':
    create_flags()
    tfv1.app.run(main)
