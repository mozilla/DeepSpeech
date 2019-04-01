#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import os
import sys

log_level_index = sys.argv.index('--log_level') + 1 if '--log_level' in sys.argv else 0
os.environ['TF_CPP_MIN_LOG_LEVEL'] = sys.argv[log_level_index] if log_level_index > 0 and log_level_index < len(sys.argv) else '3'

import time
import evaluate
import numpy as np
import progressbar
import shutil
import tensorflow as tf

from ds_ctcdecoder import ctc_beam_search_decoder, Scorer
from six.moves import zip, range
from tensorflow.python.tools import freeze_graph
from util.audio import audiofile_to_input_vector
from util.config import Config, initialize_globals
from util.feeding import DataSet, ModelFeeder
from util.flags import create_flags, FLAGS
from util.logging import log_info, log_error, log_debug, log_warn
from util.preprocess import preprocess


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
        var = tf.get_variable(name=name, shape=shape, initializer=initializer)
    return var


def BiRNN(batch_x, seq_length, dropout, reuse=False, batch_size=None, n_steps=-1, previous_state=None, tflite=False):
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
    layers = {}

    # Input shape: [batch_size, n_steps, n_input + 2*n_input*n_context]
    if not batch_size:
        batch_size = tf.shape(batch_x)[0]

    # Reshaping `batch_x` to a tensor with shape `[n_steps*batch_size, n_input + 2*n_input*n_context]`.
    # This is done to prepare the batch for input into the first layer which expects a tensor of rank `2`.

    # Permute n_steps and batch_size
    batch_x = tf.transpose(batch_x, [1, 0, 2, 3])
    # Reshape to prepare input for first layer
    batch_x = tf.reshape(batch_x, [-1, Config.n_input + 2*Config.n_input*Config.n_context]) # (n_steps*batch_size, n_input + 2*n_input*n_context)
    layers['input_reshaped'] = batch_x

    # The next three blocks will pass `batch_x` through three hidden layers with
    # clipped RELU activation and dropout.

    # 1st layer
    b1 = variable_on_cpu('b1', [Config.n_hidden_1], tf.zeros_initializer())
    h1 = variable_on_cpu('h1', [Config.n_input + 2*Config.n_input*Config.n_context, Config.n_hidden_1], tf.contrib.layers.xavier_initializer())
    layer_1 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(batch_x, h1), b1)), FLAGS.relu_clip)
    layer_1 = tf.nn.dropout(layer_1, rate=dropout[0])
    layers['layer_1'] = layer_1

    # 2nd layer
    b2 = variable_on_cpu('b2', [Config.n_hidden_2], tf.zeros_initializer())
    h2 = variable_on_cpu('h2', [Config.n_hidden_1, Config.n_hidden_2], tf.contrib.layers.xavier_initializer())
    layer_2 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(layer_1, h2), b2)), FLAGS.relu_clip)
    layer_2 = tf.nn.dropout(layer_2, rate=dropout[1])
    layers['layer_2'] = layer_2

    # 3rd layer
    b3 = variable_on_cpu('b3', [Config.n_hidden_3], tf.zeros_initializer())
    h3 = variable_on_cpu('h3', [Config.n_hidden_2, Config.n_hidden_3], tf.contrib.layers.xavier_initializer())
    layer_3 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(layer_2, h3), b3)), FLAGS.relu_clip)
    layer_3 = tf.nn.dropout(layer_3, rate=dropout[2])
    layers['layer_3'] = layer_3

    # Now we create the forward and backward LSTM units.
    # Both of which have inputs of length `n_cell_dim` and bias `1.0` for the forget gate of the LSTM.

    # Forward direction cell:
    if not tflite:
        fw_cell = tf.contrib.rnn.LSTMBlockFusedCell(Config.n_cell_dim, reuse=reuse)
        layers['fw_cell'] = fw_cell
    else:
        fw_cell = tf.nn.rnn_cell.LSTMCell(Config.n_cell_dim, reuse=reuse)

    # `layer_3` is now reshaped into `[n_steps, batch_size, 2*n_cell_dim]`,
    # as the LSTM RNN expects its input to be of shape `[max_time, batch_size, input_size]`.
    layer_3 = tf.reshape(layer_3, [n_steps, batch_size, Config.n_hidden_3])
    if tflite:
        # Generated StridedSlice, not supported by NNAPI
        #n_layer_3 = []
        #for l in range(layer_3.shape[0]):
        #    n_layer_3.append(layer_3[l])
        #layer_3 = n_layer_3

        # Unstack/Unpack is not supported by NNAPI
        layer_3 = tf.unstack(layer_3, n_steps)

    # We parametrize the RNN implementation as the training and inference graph
    # need to do different things here.
    if not tflite:
        output, output_state = fw_cell(inputs=layer_3, dtype=tf.float32, sequence_length=seq_length, initial_state=previous_state)
    else:
        output, output_state = tf.nn.static_rnn(fw_cell, layer_3, previous_state, tf.float32)
        output = tf.concat(output, 0)

    # Reshape output from a tensor of shape [n_steps, batch_size, n_cell_dim]
    # to a tensor of shape [n_steps*batch_size, n_cell_dim]
    output = tf.reshape(output, [-1, Config.n_cell_dim])
    layers['rnn_output'] = output
    layers['rnn_output_state'] = output_state

    # Now we feed `output` to the fifth hidden layer with clipped RELU activation and dropout
    b5 = variable_on_cpu('b5', [Config.n_hidden_5], tf.zeros_initializer())
    h5 = variable_on_cpu('h5', [Config.n_cell_dim, Config.n_hidden_5], tf.contrib.layers.xavier_initializer())
    layer_5 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(output, h5), b5)), FLAGS.relu_clip)
    layer_5 = tf.nn.dropout(layer_5, rate=dropout[5])
    layers['layer_5'] = layer_5

    # Now we apply the weight matrix `h6` and bias `b6` to the output of `layer_5`
    # creating `n_classes` dimensional vectors, the logits.
    b6 = variable_on_cpu('b6', [Config.n_hidden_6], tf.zeros_initializer())
    h6 = variable_on_cpu('h6', [Config.n_hidden_5, Config.n_hidden_6], tf.contrib.layers.xavier_initializer())
    layer_6 = tf.add(tf.matmul(layer_5, h6), b6)
    layers['layer_6'] = layer_6

    # Finally we reshape layer_6 from a tensor of shape [n_steps*batch_size, n_hidden_6]
    # to the slightly more useful shape [n_steps, batch_size, n_hidden_6].
    # Note, that this differs from the input in that it is time-major.
    layer_6 = tf.reshape(layer_6, [n_steps, batch_size, Config.n_hidden_6], name="raw_logits")
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

def calculate_mean_edit_distance_and_loss(model_feeder, tower, dropout, reuse):
    r'''
    This routine beam search decodes a mini-batch and calculates the loss and mean edit distance.
    Next to total and average loss it returns the mean edit distance,
    the decoded result and the batch's original Y.
    '''
    # Obtain the next batch of data
    batch_x, batch_seq_len, batch_y = model_feeder.next_batch(tower)

    # Calculate the logits of the batch using BiRNN
    logits, _ = BiRNN(batch_x, batch_seq_len, dropout, reuse)

    # Compute the CTC loss using TensorFlow's `ctc_loss`
    total_loss = tf.nn.ctc_loss(labels=batch_y, inputs=logits, sequence_length=batch_seq_len)

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

def get_tower_results(model_feeder, optimizer, dropout_rates):
    r'''
    With this preliminary step out of the way, we can for each GPU introduce a
    tower for which's batch we calculate and return the optimization gradients
    and the average loss across towers.
    '''
    # To calculate the mean of the losses
    tower_avg_losses = []

    # Tower gradients to return
    tower_gradients = []

    with tf.variable_scope(tf.get_variable_scope()):
        # Loop over available_devices
        for i in range(len(Config.available_devices)):
            # Execute operations of tower i on device i
            device = Config.available_devices[i]
            with tf.device(device):
                # Create a scope for all operations of tower i
                with tf.name_scope('tower_%d' % i) as scope:
                    # Calculate the avg_loss and mean_edit_distance and retrieve the decoded
                    # batch along with the original batch's labels (Y) of this tower
                    avg_loss = calculate_mean_edit_distance_and_loss(model_feeder, i, dropout_rates, reuse=i>0)

                    # Allow for variables to be re-used by the next tower
                    tf.get_variable_scope().reuse_variables()

                    # Retain tower's avg losses
                    tower_avg_losses.append(avg_loss)

                    # Compute gradients for model parameters using tower's mini-batch
                    gradients = optimizer.compute_gradients(avg_loss)

                    # Retain tower's gradients
                    tower_gradients.append(gradients)


    avg_loss_across_towers = tf.reduce_mean(tower_avg_losses, 0)

    tf.summary.scalar(name='step_loss', tensor=avg_loss_across_towers, collections=['step_summaries'])

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


# Helpers
# =======


class SampleIndex:
    def __init__(self, index=0):
        self.index = index

    def inc(self, old_index):
        self.index += 1
        return self.index


def try_loading(session, saver, checkpoint_filename, caption):
    try:
        checkpoint = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir, checkpoint_filename)
        if not checkpoint:
            return False
        checkpoint_path = checkpoint.model_checkpoint_path
        saver.restore(session, checkpoint_path)
        log_info('Restored model from %s checkpoint at %s' % (caption, checkpoint_path))
        return True
    except tf.errors.InvalidArgumentError as e:
        log_error(str(e))
        log_error('The checkpoint in {0} does not match the shapes of the model.'
                  ' Did you change alphabet.txt or the --n_hidden parameter'
                  ' between train runs using the same checkpoint dir? Try moving'
                  ' or removing the contents of {0}.'.format(checkpoint_path))
        sys.exit(1)
    except:
        return False


def train():
    r'''
    Trains the network on a given server of a cluster.
    If no server provided, it performs single process training.
    '''

    # Reading training set
    train_index = SampleIndex()

    train_data = preprocess(FLAGS.train_files.split(','),
                            FLAGS.train_batch_size,
                            Config.n_input,
                            Config.n_context,
                            Config.alphabet,
                            hdf5_cache_path=FLAGS.train_cached_features_path)

    train_set = DataSet(train_data,
                        FLAGS.train_batch_size,
                        limit=FLAGS.limit_train,
                        next_index=train_index.inc)

    # Reading validation set
    dev_index = SampleIndex()

    dev_data = preprocess(FLAGS.dev_files.split(','),
                          FLAGS.dev_batch_size,
                          Config.n_input,
                          Config.n_context,
                          Config.alphabet,
                          hdf5_cache_path=FLAGS.dev_cached_features_path)

    dev_set = DataSet(dev_data,
                      FLAGS.dev_batch_size,
                      limit=FLAGS.limit_dev,
                      next_index=dev_index.inc)

    # Combining all sets to a multi set model feeder
    model_feeder = ModelFeeder(train_set,
                               dev_set,
                               Config.n_input,
                               Config.n_context,
                               Config.alphabet,
                               tower_feeder_count=len(Config.available_devices))

    # Dropout
    dropout_rates = [tf.placeholder(tf.float32, name='dropout_{}'.format(i)) for i in range(6)]
    dropout_feed_dict = {
        dropout_rates[0]: FLAGS.dropout_rate,
        dropout_rates[1]: FLAGS.dropout_rate2,
        dropout_rates[2]: FLAGS.dropout_rate3,
        dropout_rates[3]: FLAGS.dropout_rate4,
        dropout_rates[4]: FLAGS.dropout_rate5,
        dropout_rates[5]: FLAGS.dropout_rate6,
    }
    no_dropout_feed_dict = {
        dropout_rates[0]: 0.,
        dropout_rates[1]: 0.,
        dropout_rates[2]: 0.,
        dropout_rates[3]: 0.,
        dropout_rates[4]: 0.,
        dropout_rates[5]: 0.,
    }

    # Building the graph
    optimizer = create_optimizer()
    gradients, loss = get_tower_results(model_feeder, optimizer, dropout_rates)
    # Average tower gradients across GPUs
    avg_tower_gradients = average_gradients(gradients)
    log_grads_and_vars(avg_tower_gradients)
    # global_step is automagically incremented by the optimizer
    global_step = tf.Variable(0, trainable=False, name='global_step')
    apply_gradient_op = optimizer.apply_gradients(avg_tower_gradients, global_step=global_step)

    # Summaries
    step_summaries_op = tf.summary.merge_all('step_summaries')
    step_summary_writers = {
        'train': tf.summary.FileWriter(os.path.join(FLAGS.summary_dir, 'train'), max_queue=120),
        'dev': tf.summary.FileWriter(os.path.join(FLAGS.summary_dir, 'dev'), max_queue=120)
    }

    # Checkpointing
    checkpoint_saver = tf.train.Saver(max_to_keep=FLAGS.max_to_keep)
    checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'train')
    checkpoint_filename = 'checkpoint'

    best_dev_saver = tf.train.Saver(max_to_keep=1)
    best_dev_path = os.path.join(FLAGS.checkpoint_dir, 'best_dev')
    best_dev_filename = 'best_dev_checkpoint'

    initializer = tf.global_variables_initializer()

    with tf.Session(config=Config.session_config) as session:
        log_debug('Session opened.')
        tf.get_default_graph().finalize()

        # Loading or initializing
        loaded = False
        if FLAGS.load in ['auto', 'last']:
            loaded = try_loading(session, checkpoint_saver, checkpoint_filename, 'most recent epoch')
        if not loaded and FLAGS.load in ['auto', 'best']:
            loaded = try_loading(session, best_dev_saver, best_dev_filename, 'best validation')
        if not loaded:
            if FLAGS.load in ['auto', 'init']:
                log_info('Initializing...')
                session.run(initializer)
            else:
                log_error('Unable to load %s model from specified checkpoint dir'
                          ' - consider using load option "auto" or "init".' % FLAGS.load)
                sys.exit(1)

        # Retrieving global_step from restored model and setting training parameters accordingly
        model_feeder.set_data_set(no_dropout_feed_dict, train_set)
        step = session.run(global_step, feed_dict=no_dropout_feed_dict)
        num_gpus = len(Config.available_devices)
        steps_per_epoch = max(1, train_set.total_batches // num_gpus)
        steps_trained = step % steps_per_epoch
        current_epoch = step // steps_per_epoch
        target_epoch = current_epoch + abs(FLAGS.epoch) if FLAGS.epoch < 0 else FLAGS.epoch
        train_index.index = steps_trained * num_gpus

        log_debug('step: %d' % step)
        log_debug('epoch: %d' % current_epoch)
        log_debug('target epoch: %d' % target_epoch)
        log_debug('steps per epoch: %d' % steps_per_epoch)
        log_debug('batches per step (GPUs): %d' % num_gpus)
        log_debug('number of batches in train set: %d' % train_set.total_batches)
        log_debug('number of batches already trained in epoch: %d' % train_index.index)

        def run_set(set_name):
            data_set = getattr(model_feeder, set_name)
            is_train = set_name == 'train'
            train_op = apply_gradient_op if is_train else []
            feed_dict = dropout_feed_dict if is_train else no_dropout_feed_dict
            model_feeder.set_data_set(feed_dict, data_set)
            total_loss = 0.0
            step_summary_writer = step_summary_writers.get(set_name)
            num_steps = max(1, data_set.total_batches // num_gpus)
            checkpoint_time = time.time()
            if FLAGS.show_progressbar:
                pbar = progressbar.ProgressBar(max_value=num_steps, redirect_stdout=True).start()
            # Batch loop
            for step_index in range(steps_trained, num_steps):
                if coord.should_stop():
                    break
                _, current_step, batch_loss, step_summary = \
                    session.run([train_op, global_step, loss, step_summaries_op],
                                feed_dict=feed_dict)
                total_loss += batch_loss
                step_summary_writer.add_summary(step_summary, current_step)
                if FLAGS.show_progressbar:
                    pbar.update(step_index + 1, force=True)
                if is_train and FLAGS.checkpoint_secs > 0 and time.time() - checkpoint_time > FLAGS.checkpoint_secs:
                    checkpoint_saver.save(session, checkpoint_path, global_step=current_step)
                    checkpoint_time = time.time()
            if FLAGS.show_progressbar:
                pbar.finish()
            return total_loss / num_steps

        if target_epoch > current_epoch:
            log_info('STARTING Optimization')
            best_dev_loss = float('inf')
            dev_losses = []
            coord = tf.train.Coordinator()
            with coord.stop_on_exception():
                log_debug('Starting queue runners...')
                model_feeder.start_queue_threads(session, coord=coord)
                log_debug('Queue runners started.')
                # Epoch loop
                for current_epoch in range(current_epoch, target_epoch):
                    # Training
                    if coord.should_stop():
                        break
                    log_info('Training epoch %d ...' % current_epoch)
                    train_loss = run_set('train')
                    log_info('Finished training epoch %d - loss: %f' % (current_epoch, train_loss))
                    checkpoint_saver.save(session, checkpoint_path, global_step=global_step)
                    steps_trained = 0
                    # Validation
                    log_info('Validating epoch %d ...' % current_epoch)
                    dev_loss = run_set('dev')
                    dev_losses.append(dev_loss)
                    log_info('Finished validating epoch %d - loss: %f' % (current_epoch, dev_loss))
                    if dev_loss < best_dev_loss:
                        best_dev_loss = dev_loss
                        save_path = best_dev_saver.save(session, best_dev_path, latest_filename=best_dev_filename)
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
                log_debug('Closing queues...')
                coord.request_stop()
                model_feeder.close_queues(session)
                log_debug('Queues closed.')
        else:
            log_info('Target epoch already reached - skipped training.')
    log_debug('Session closed.')


def test():
    # Reading test set
    test_data = preprocess(FLAGS.test_files.split(','),
                           FLAGS.test_batch_size,
                           Config.n_input,
                           Config.n_context,
                           Config.alphabet,
                           hdf5_cache_path=FLAGS.test_cached_features_path)

    graph = create_inference_graph(batch_size=FLAGS.test_batch_size, n_steps=-1)
    evaluate.evaluate(test_data, graph)


def create_inference_graph(batch_size=1, n_steps=16, tflite=False):
    batch_size = batch_size if batch_size > 0 else None
    # Input tensor will be of shape [batch_size, n_steps, 2*n_context+1, n_input]
    input_tensor = tf.placeholder(tf.float32, [batch_size, n_steps if n_steps > 0 else None, 2*Config.n_context+1, Config.n_input], name='input_node')
    seq_length = tf.placeholder(tf.int32, [batch_size], name='input_lengths')

    if batch_size <= 0:
        # no state management since n_step is expected to be dynamic too (see below)
        previous_state = previous_state_c = previous_state_h = None
    else:
        if not tflite:
            previous_state_c = variable_on_cpu('previous_state_c', [batch_size, Config.n_cell_dim], initializer=None)
            previous_state_h = variable_on_cpu('previous_state_h', [batch_size, Config.n_cell_dim], initializer=None)
        else:
            previous_state_c = tf.placeholder(tf.float32, [batch_size, Config.n_cell_dim], name='previous_state_c')
            previous_state_h = tf.placeholder(tf.float32, [batch_size, Config.n_cell_dim], name='previous_state_h')

        previous_state = tf.contrib.rnn.LSTMStateTuple(previous_state_c, previous_state_h)

    no_dropout = [0.0] * 6

    logits, layers = BiRNN(batch_x=input_tensor,
                           seq_length=seq_length if FLAGS.use_seq_length else None,
                           dropout=no_dropout,
                           batch_size=batch_size,
                           n_steps=n_steps,
                           previous_state=previous_state,
                           tflite=tflite)

    # TF Lite runtime will check that input dimensions are 1, 2 or 4
    # by default we get 3, the middle one being batch_size which is forced to
    # one on inference graph, so remove that dimension
    if tflite:
        logits = tf.squeeze(logits, [1])

    # Apply softmax for CTC decoder
    logits = tf.nn.softmax(logits)

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
                'outputs': tf.identity(logits, name='logits'),
            },
            layers
        )

    new_state_c, new_state_h = layers['rnn_output_state']
    if not tflite:
        zero_state = tf.zeros([batch_size, Config.n_cell_dim], tf.float32)
        initialize_c = tf.assign(previous_state_c, zero_state)
        initialize_h = tf.assign(previous_state_h, zero_state)
        initialize_state = tf.group(initialize_c, initialize_h, name='initialize_state')
        with tf.control_dependencies([tf.assign(previous_state_c, new_state_c), tf.assign(previous_state_h, new_state_h)]):
            logits = tf.identity(logits, name='logits')

        return (
            {
                'input': input_tensor,
                'input_lengths': seq_length,
            },
            {
                'outputs': logits,
                'initialize_state': initialize_state,
            },
            layers
        )
    else:
        logits = tf.identity(logits, name='logits')
        new_state_c = tf.identity(new_state_c, name='new_state_c')
        new_state_h = tf.identity(new_state_h, name='new_state_h')

        return (
            {
                'input': input_tensor,
                'previous_state_c': previous_state_c,
                'previous_state_h': previous_state_h,
            },
            {
                'outputs': logits,
                'new_state_c': new_state_c,
                'new_state_h': new_state_h,
            },
            layers
        )


def export():
    r'''
    Restores the trained variables into a simpler graph that will be exported for serving.
    '''
    log_info('Exporting the model...')
    with tf.device('/cpu:0'):
        from tensorflow.python.framework.ops import Tensor, Operation

        tf.reset_default_graph()
        session = tf.Session(config=Config.session_config)

        inputs, outputs, _ = create_inference_graph(batch_size=FLAGS.export_batch_size, n_steps=FLAGS.n_steps, tflite=FLAGS.export_tflite)
        input_names = ",".join(tensor.op.name for tensor in inputs.values())
        output_names_tensors = [ tensor.op.name for tensor in outputs.values() if isinstance(tensor, Tensor) ]
        output_names_ops = [ tensor.name for tensor in outputs.values() if isinstance(tensor, Operation) ]
        output_names = ",".join(output_names_tensors + output_names_ops)
        input_shapes = ":".join(",".join(map(str, tensor.shape)) for tensor in inputs.values())

        if not FLAGS.export_tflite:
            mapping = {v.op.name: v for v in tf.global_variables() if not v.op.name.startswith('previous_state_')}
        else:
            # Create a saver using variables from the above newly created graph
            def fixup(name):
                if name.startswith('rnn/lstm_cell/'):
                    return name.replace('rnn/lstm_cell/', 'lstm_fused_cell/')
                return name

            mapping = {fixup(v.op.name): v for v in tf.global_variables()}

        saver = tf.train.Saver(mapping)

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

            def do_graph_freeze(output_file=None, output_node_names=None, variables_blacklist=None):
                return freeze_graph.freeze_graph_with_def_protos(
                    input_graph_def=session.graph_def,
                    input_saver_def=saver.as_saver_def(),
                    input_checkpoint=checkpoint_path,
                    output_node_names=output_node_names,
                    restore_op_name=None,
                    filename_tensor_name=None,
                    output_graph=output_file,
                    clear_devices=False,
                    variable_names_blacklist=variables_blacklist,
                    initializer_nodes='')

            if not FLAGS.export_tflite:
                do_graph_freeze(output_file=output_graph_path, output_node_names=output_names, variables_blacklist='previous_state_c,previous_state_h')
            else:
                frozen_graph = do_graph_freeze(output_node_names=output_names, variables_blacklist='')
                output_tflite_path = os.path.join(FLAGS.export_dir, output_filename.replace('.pb', '.tflite'))

                converter = tf.lite.TFLiteConverter(frozen_graph, input_tensors=inputs.values(), output_tensors=outputs.values())
                converter.post_training_quantize = True
                tflite_model = converter.convert()

                with open(output_tflite_path, 'wb') as fout:
                    fout.write(tflite_model)

                log_info('Exported model for TF Lite engine as {}'.format(os.path.basename(output_tflite_path)))

            log_info('Models exported at %s' % (FLAGS.export_dir))
        except RuntimeError as e:
            log_error(str(e))

def do_single_file_inference(input_file_path):
    with tf.Session(config=Config.session_config) as session:
        inputs, outputs, _ = create_inference_graph(batch_size=1, n_steps=-1)

        # Create a saver using variables from the above newly created graph
        mapping = {v.op.name: v for v in tf.global_variables() if not v.op.name.startswith('previous_state_')}
        saver = tf.train.Saver(mapping)

        # Restore variables from training checkpoint
        # TODO: This restores the most recent checkpoint, but if we use validation to counteract
        #       over-fitting, we may want to restore an earlier checkpoint.
        checkpoint = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if not checkpoint:
            log_error('Checkpoint directory ({}) does not contain a valid checkpoint state.'.format(FLAGS.checkpoint_dir))
            exit(1)

        checkpoint_path = checkpoint.model_checkpoint_path
        saver.restore(session, checkpoint_path)
        session.run(outputs['initialize_state'])

        features = audiofile_to_input_vector(input_file_path, Config.n_input, Config.n_context)
        num_strides = len(features) - (Config.n_context * 2)

        # Create a view into the array with overlapping strides of size
        # numcontext (past) + 1 (present) + numcontext (future)
        window_size = 2*Config.n_context+1
        features = np.lib.stride_tricks.as_strided(
            features,
            (num_strides, window_size, Config.n_input),
            (features.strides[0], features.strides[0], features.strides[1]),
            writeable=False)

        logits = session.run(outputs['outputs'], feed_dict = {
            inputs['input']: [features],
            inputs['input_lengths']: [num_strides],
        })

        logits = np.squeeze(logits)

        scorer = Scorer(FLAGS.lm_alpha, FLAGS.lm_beta,
                        FLAGS.lm_binary_path, FLAGS.lm_trie_path,
                        Config.alphabet)
        decoded = ctc_beam_search_decoder(logits, Config.alphabet, FLAGS.beam_width, scorer=scorer)
        # Print highest probability result
        print(decoded[0][1])


def main(_):
    initialize_globals()

    if FLAGS.train:
        with tf.Graph().as_default():
            tf.set_random_seed(FLAGS.random_seed)
            train()

    if FLAGS.test:
        with tf.Graph().as_default():
            test()

    if FLAGS.export_dir:
        export()

    if len(FLAGS.one_shot_infer):
        do_single_file_inference(FLAGS.one_shot_infer)

if __name__ == '__main__' :
    create_flags()
    tf.app.run(main)
