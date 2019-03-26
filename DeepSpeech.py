#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

NUM_LANGS=2

import os
import sys

log_level_index = sys.argv.index('--log_level') + 1 if '--log_level' in sys.argv else 0
os.environ['TF_CPP_MIN_LOG_LEVEL'] = sys.argv[log_level_index] if log_level_index > 0 and log_level_index < len(sys.argv) else '3'

import evaluate
import numpy as np
import progressbar
import shutil
import tensorflow as tf
import traceback
import pandas
from ds_ctcdecoder import ctc_beam_search_decoder, Scorer
from six.moves import zip, range
from tensorflow.python.tools import freeze_graph
from util.audio import audiofile_to_input_vector
from util.config import Config, initialize_globals
from util.coordinator import TrainingCoordinator
from util.feeding import DataSet, ModelFeeder
from util.flags import create_flags, FLAGS
from util.logging import log_info, log_error, log_debug, log_warn
from util.preprocess import preprocess
from util.text import Alphabet


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
        device = Config.worker_device
    else:
        device = tf.train.replica_device_setter(worker_device=Config.worker_device, cluster=Config.cluster)

    with tf.device(device):
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
    b1 = variable_on_worker_level('b1', [Config.n_hidden_1], tf.zeros_initializer())
    h1 = variable_on_worker_level('h1', [Config.n_input + 2*Config.n_input*Config.n_context, Config.n_hidden_1], tf.contrib.layers.xavier_initializer())
    layer_1 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(batch_x, h1), b1)), FLAGS.relu_clip)
    layer_1 = tf.nn.dropout(layer_1, rate=dropout[0])
    layers['layer_1'] = layer_1

    # 2nd layer
    b2 = variable_on_worker_level('b2', [Config.n_hidden_2], tf.zeros_initializer())
    h2 = variable_on_worker_level('h2', [Config.n_hidden_1, Config.n_hidden_2], tf.contrib.layers.xavier_initializer())
    layer_2 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(layer_1, h2), b2)), FLAGS.relu_clip)
    layer_2 = tf.nn.dropout(layer_2, rate=dropout[1])
    layers['layer_2'] = layer_2

    # 3rd layer
    b3 = variable_on_worker_level('b3', [Config.n_hidden_3], tf.zeros_initializer())
    h3 = variable_on_worker_level('h3', [Config.n_hidden_2, Config.n_hidden_3], tf.contrib.layers.xavier_initializer())
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
    b5 = variable_on_worker_level('b5', [Config.n_hidden_5], tf.zeros_initializer())
    h5 = variable_on_worker_level('h5', [Config.n_cell_dim, Config.n_hidden_5], tf.contrib.layers.xavier_initializer())
    layer_5 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(output, h5), b5)), FLAGS.relu_clip)
    layer_5 = tf.nn.dropout(layer_5, rate=dropout[5])
    layers['layer_5'] = layer_5

    # Now we apply the weight matrix `h6` and bias `b6` to the output of `layer_5`
    # creating `n_classes` dimensional vectors, the logits.
    b6 = variable_on_worker_level('b6', [NUM_LANGS], tf.zeros_initializer())
    h6 = variable_on_worker_level('h6', [Config.n_hidden_5, NUM_LANGS], tf.contrib.layers.xavier_initializer())
    layer_6 = tf.add(tf.matmul(layer_5, h6), b6)
    layers['layer_6'] = layer_6

    # Finally we reshape layer_6 from a tensor of shape [n_steps*batch_size, n_hidden_6]
    # to the slightly more useful shape [n_steps, batch_size, n_hidden_6].
    # Note, that this differs from the input in that it is time-major.
    layer_6 = tf.reshape(layer_6, [n_steps, batch_size, NUM_LANGS], name="raw_logits")
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

    # SPECIAL FOR LANG-ID
    
    batch_shape = tf.shape(batch_x)
    batch_size = batch_shape[0]
    n_steps = batch_shape[1]
    output = logits
    # permute time and batch
    output = tf.reshape(output, [n_steps, batch_size, -1])
    output = tf.transpose(output, [1, 0, 2])
    output = tf.reshape(output, [batch_size, n_steps, -1])
    # mean over time steps
    output = tf.reduce_sum(output, axis=1) / tf.cast(batch_seq_len, tf.float32)
    total_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.cast(batch_y, tf.float32), logits=output
    )

    # Compute the CTC loss using TensorFlow's `ctc_loss`
    #total_loss = tf.nn.ctc_loss(labels=batch_y, inputs=logits, sequence_length=batch_seq_len)

    output_probs = tf.sigmoid(output)
    prediction = tf.cast(output_probs > 0.5, tf.int32)
    eq = tf.cast(tf.equal(prediction, batch_y), tf.int32)
    sum1 = tf.reduce_sum(eq)
    sum2 = tf.reduce_sum(tf.ones_like(prediction))
    accuracy = sum1 / sum2
    
    # Calculate the average loss across the batch
    avg_loss = tf.reduce_mean(total_loss)

    # Finally we return the average loss
    return avg_loss, accuracy


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
    tower_avg_acc = []

    # Tower gradients to return
    tower_gradients = []

    with tf.variable_scope(tf.get_variable_scope()):
        # Loop over available_devices
        for i in range(len(Config.available_devices)):
            # Execute operations of tower i on device i
            if len(FLAGS.ps_hosts) == 0:
                device = Config.available_devices[i]
            else:
                device = tf.train.replica_device_setter(worker_device=Config.available_devices[i], cluster=Config.cluster)
            with tf.device(device):
                # Create a scope for all operations of tower i
                with tf.name_scope('tower_%d' % i) as scope:
                    # Calculate the avg_loss and mean_edit_distance and retrieve the decoded
                    # batch along with the original batch's labels (Y) of this tower
                    avg_loss, acc  = calculate_mean_edit_distance_and_loss(model_feeder, i, dropout_rates, reuse=i>0)

                    # Allow for variables to be re-used by the next tower
                    tf.get_variable_scope().reuse_variables()

                    # Retain tower's avg losses
                    tower_avg_losses.append(avg_loss)
                    tower_avg_acc.append(acc)
                    
                    # Compute gradients for model parameters using tower's mini-batch
                    gradients = optimizer.compute_gradients(avg_loss)

                    # Retain tower's gradients
                    tower_gradients.append(gradients)


    avg_loss_across_towers = tf.reduce_mean(tower_avg_losses, 0)
    avg_acc_across_towers = tf.reduce_mean(tower_avg_acc, 0)
    
    tf.summary.scalar(name='step_loss', tensor=avg_loss_across_towers, collections=['step_summaries'])

    # Return gradients and the average loss
    return tower_gradients, avg_loss_across_towers, avg_acc_across_towers


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

def send_token_to_ps(session, kill=False):
    # Sending our token (the task_index as a debug opportunity) to each parameter server.
    # kill switch tokens are negative and decremented by 1 to deal with task_index 0
    token = -FLAGS.task_index-1 if kill else FLAGS.task_index
    kind = 'kill switch' if kill else 'stop'
    for index, enqueue in enumerate(Config.done_enqueues):
        log_debug('Sending %s token to ps %d...' % (kind, index))
        session.run(enqueue, feed_dict={ Config.token_placeholder: token })
        log_debug('Sent %s token to ps %d.' % (kind, index))


def train(server=None):
    r'''
    Trains the network on a given server of a cluster.
    If no server provided, it performs single process training.
    '''

    # Initializing and starting the training coordinator
    coord = TrainingCoordinator(Config.is_chief)
    coord.start()

    # Create a variable to hold the global_step.
    # It will automagically get incremented by the optimizer.
    global_step = tf.Variable(0, trainable=False, name='global_step')

    dropout_rates = [tf.placeholder(tf.float32, name='dropout_{}'.format(i)) for i in range(6)]

    # Reading training set
    train_welsh = preprocess(FLAGS.train_welsh.split(','),
                            FLAGS.train_batch_size,
                            Config.n_input,
                            Config.n_context,
                            Config.alphabet,
                            hdf5_cache_path=FLAGS.train_cached_features_path,
                            file_type='welsh')

    train_kyrgyz = preprocess(FLAGS.train_kyrgyz.split(','),
                                 FLAGS.train_batch_size,
                                 Config.n_input,
                                 Config.n_context,
                                 Config.alphabet,
                                 file_type='kyrgyz')
    
    # train_set = DataSet(train_data,
    train_set = DataSet(pandas.concat((train_welsh, train_kyrgyz)),
                        FLAGS.train_batch_size,
                        limit=FLAGS.limit_train,
                        next_index=lambda i: coord.get_next_index('train'))

    

    # Reading deving set
    dev_welsh = preprocess(FLAGS.dev_welsh.split(','),
                            FLAGS.dev_batch_size,
                            Config.n_input,
                            Config.n_context,
                            Config.alphabet,
                            hdf5_cache_path=FLAGS.dev_cached_features_path,
                            file_type='welsh')

    dev_kyrgyz = preprocess(FLAGS.dev_kyrgyz.split(','),
                                 FLAGS.dev_batch_size,
                                 Config.n_input,
                                 Config.n_context,
                                 Config.alphabet,
                                 file_type='kyrgyz')
    
    # dev_set = DataSet(dev_data,
    dev_set = DataSet(pandas.concat((dev_welsh, dev_kyrgyz)),
                        FLAGS.dev_batch_size,
                        limit=FLAGS.limit_dev,
                        next_index=lambda i: coord.get_next_index('dev'))

    

    # Combining all sets to a multi set model feeder
    model_feeder = ModelFeeder(train_set,
                               dev_set,
                               Config.n_input,
                               Config.n_context,
                               Config.alphabet,
                               tower_feeder_count=len(Config.available_devices))

    # Create the optimizer
    optimizer = create_optimizer()

    # Synchronous distributed training is facilitated by a special proxy-optimizer
    if not server is None:
        optimizer = tf.train.SyncReplicasOptimizer(optimizer,
                                                   replicas_to_aggregate=FLAGS.replicas_to_agg,
                                                   total_num_replicas=FLAGS.replicas)

    # Get the data_set specific graph end-points
    gradients, loss, acc = get_tower_results(model_feeder, optimizer, dropout_rates)

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
        'dev': tf.summary.FileWriter(os.path.join(FLAGS.summary_dir, 'dev'), max_queue=120)
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
        hooks.append(optimizer.make_session_run_hook(Config.is_chief))

    # Hook to save TensorBoard summaries
    if FLAGS.summary_secs > 0:
        hooks.append(tf.train.SummarySaverHook(save_secs=FLAGS.summary_secs, output_dir=FLAGS.summary_dir, summary_op=merge_all_summaries_op))

    # Hook wih number of checkpoint files to save in checkpoint_dir
    if FLAGS.train and FLAGS.max_to_keep > 0:
        saver = tf.train.Saver(max_to_keep=FLAGS.max_to_keep)
        hooks.append(tf.train.CheckpointSaverHook(checkpoint_dir=FLAGS.checkpoint_dir, save_secs=FLAGS.checkpoint_secs, saver=saver))

    no_dropout_feed_dict = {
        dropout_rates[0]: 0.,
        dropout_rates[1]: 0.,
        dropout_rates[2]: 0.,
        dropout_rates[3]: 0.,
        dropout_rates[4]: 0.,
        dropout_rates[5]: 0.,
    }

    # Progress Bar
    def update_progressbar(set_name):
        if not hasattr(update_progressbar, 'current_set_name'):
            update_progressbar.current_set_name = None

        if (update_progressbar.current_set_name != set_name or
            update_progressbar.current_job_index == update_progressbar.total_jobs):

            # finish prev pbar if it exists
            if hasattr(update_progressbar, 'pbar') and update_progressbar.pbar:
                update_progressbar.pbar.finish()

            update_progressbar.total_jobs = None
            update_progressbar.current_job_index = 0

            current_epoch = coord._epoch-1

            if set_name == "train":
                log_info('Training epoch %i...' % current_epoch)
                update_progressbar.total_jobs = coord._num_jobs_train
            else:
                log_info('Validating epoch %i...' % current_epoch)
                update_progressbar.total_jobs = coord._num_jobs_dev

            # recreate pbar
            update_progressbar.pbar = progressbar.ProgressBar(max_value=update_progressbar.total_jobs,
                                                              redirect_stdout=True).start()

            update_progressbar.current_set_name = set_name

        if update_progressbar.pbar:
            update_progressbar.pbar.update(update_progressbar.current_job_index+1, force=True)

        update_progressbar.current_job_index += 1

    # Initialize update_progressbar()'s child fields to safe values
    update_progressbar.pbar = None

    # The MonitoredTrainingSession takes care of session initialization,
    # restoring from a checkpoint, saving to a checkpoint, and closing when done
    # or an error occurs.
    try:
        with tf.train.MonitoredTrainingSession(master='' if server is None else server.target,
                                               is_chief=Config.is_chief,
                                               hooks=hooks,
                                               checkpoint_dir=FLAGS.checkpoint_dir,
                                               save_checkpoint_secs=None, # already taken care of by a hook
                                               log_step_count_steps=0, # disable logging of steps/s to avoid TF warning in validation sets
                                               config=Config.session_config) as session:
            tf.get_default_graph().finalize()

            try:
                if Config.is_chief:
                    # Retrieving global_step from the (potentially restored) model
                    model_feeder.set_data_set(no_dropout_feed_dict, model_feeder.train)
                    step = session.run(global_step, feed_dict=no_dropout_feed_dict)
                    coord.start_coordination(model_feeder, step)

                # Get the first job
                job = coord.get_job()

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
                    total_acc = 0.0

                    # Setting the training operation in case of training requested
                    train_op = apply_gradient_op if is_train else []

                    # So far the only extra parameter is the feed_dict
                    extra_params = { 'feed_dict': feed_dict }

                    step_summary_writer = step_summary_writers.get(job.set_name)

                    # Loop over the batches
                    for job_step in range(job.steps):
                        if session.should_stop():
                            break

                        log_debug('Starting batch...')
                        # Compute the batch
                        _, current_step, batch_loss, batch_acc, step_summary = session.run([train_op, global_step, loss, step_summaries_op], **extra_params)

                        # Log step summaries
                        step_summary_writer.add_summary(step_summary, current_step)

                        # Uncomment the next line for debugging race conditions / distributed TF
                        log_debug('Finished batch step %d.' % current_step)

                        # Add batch to loss
                        total_loss += batch_loss
                        total_acc += batch_acc

                    # Gathering job results
                    job.loss = total_loss / job.steps
                    job.acc = total_acc / job.steps

                    # Display progressbar
                    if FLAGS.show_progressbar:
                        update_progressbar(job.set_name)

                    # Send the current job to coordinator and receive the next one
                    log_debug('Sending %s...' % job)
                    job = coord.next_job(job)

                if update_progressbar.pbar:
                    update_progressbar.pbar.finish()

            except Exception as e:
                log_error(str(e))
                traceback.print_exc()
                # Calling all hook's end() methods to end blocking calls
                for hook in hooks:
                    hook.end(session)
                # Only chief has a SyncReplicasOptimizer queue runner that needs to be stopped for unblocking process exit.
                # A rather graceful way to do this is by stopping the ps.
                # Only one party can send it w/o failing.
                if Config.is_chief:
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

    # Stopping the coordinator
    coord.stop()


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
            previous_state_c = variable_on_worker_level('previous_state_c', [batch_size, Config.n_cell_dim], initializer=None)
            previous_state_h = variable_on_worker_level('previous_state_h', [batch_size, Config.n_cell_dim], initializer=None)
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

    if FLAGS.train or FLAGS.test:
        if len(FLAGS.worker_hosts) == 0:
            # Only one local task: this process (default case - no cluster)
            with tf.Graph().as_default():
                tf.set_random_seed(FLAGS.random_seed)
                train()
            # Now do a final test epoch
            if FLAGS.test:
                with tf.Graph().as_default():
                    test()
            log_debug('Done.')
        else:
            # Create and start a server for the local task.
            server = tf.train.Server(Config.cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
            if FLAGS.job_name == 'ps':
                # We are a parameter server and therefore we just wait for all workers to finish
                # by waiting for their stop tokens.
                with tf.Session(server.target) as session:
                    for worker in FLAGS.worker_hosts:
                        log_debug('Waiting for stop token...')
                        token = session.run(Config.done_dequeues[FLAGS.task_index])
                        if token < 0:
                            log_debug('Got a kill switch token from worker %i.' % abs(token + 1))
                            break
                        log_debug('Got a stop token from worker %i.' % token)
                log_debug('Session closed.')

                if FLAGS.test:
                    test()
            elif FLAGS.job_name == 'worker':
                # We are a worker and therefore we have to do some work.
                # Assigns ops to the local worker by default.
                with tf.device(tf.train.replica_device_setter(
                               worker_device=Config.worker_device,
                               cluster=Config.cluster)):

                    # Do the training
                    train(server)

            log_debug('Server stopped.')

    # Are we the main process?
    if Config.is_chief:
        # Doing solo/post-processing work just on the main process...
        # Exporting the model
        if FLAGS.export_dir:
            export()

    if len(FLAGS.one_shot_infer):
        do_single_file_inference(FLAGS.one_shot_infer)

if __name__ == '__main__' :
    create_flags()
    tf.app.run(main)
