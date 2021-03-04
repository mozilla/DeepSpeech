#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import os
import sys

LOG_LEVEL_INDEX = sys.argv.index('--log_level') + 1 if '--log_level' in sys.argv else 0
DESIRED_LOG_LEVEL = sys.argv[LOG_LEVEL_INDEX] if 0 < LOG_LEVEL_INDEX < len(sys.argv) else '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = DESIRED_LOG_LEVEL

import absl.app
import numpy as np
import progressbar
import shutil
import tensorflow as tf
import tensorflow.compat.v1 as tfv1
import time

tfv1.logging.set_verbosity({
    '0': tfv1.logging.DEBUG,
    '1': tfv1.logging.INFO,
    '2': tfv1.logging.WARN,
    '3': tfv1.logging.ERROR
}.get(DESIRED_LOG_LEVEL))

from collections import defaultdict
from datetime import datetime
from ds_ctcdecoder import ctc_beam_search_decoder, Scorer
from .evaluate import evaluate
from six.moves import zip, range
from .util.config import Config, initialize_globals
from .util.checkpoints import load_or_init_graph_for_training, load_graph_for_evaluation, reload_best_checkpoint
from .util.evaluate_tools import save_samples_json
from .util.feeding import create_dataset, audio_to_features, audiofile_to_features
from .util.flags import create_flags, FLAGS
from .util.helpers import check_ctcdecoder_version, ExceptionBox
from .util.logging import create_progressbar, log_debug, log_error, log_info, log_progress, log_warn
from .util.io import open_remote, remove_remote, rmtree_remote, listdir_remote, is_remote_path, isdir_remote

check_ctcdecoder_version()

# Graph Creation
# ==============


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
    batch_x = tf.nn.conv1d(input=batch_x, filters=eye_filter, stride=1, padding='SAME')

    # Remove dummy depth dimension and reshape into [batch_size, n_windows, window_width, n_input]
    batch_x = tf.reshape(batch_x, [batch_size, -1, window_width, num_channels])

    return batch_x


class CreateOverlappingWindows(tf.keras.Model):
    def __init__(self):
        super(CreateOverlappingWindows, self).__init__()
        window_width = 2 * Config.n_context + 1
        num_channels = Config.n_input
        identity = np.eye(window_width * num_channels).reshape(
            window_width, num_channels, window_width * num_channels
        )
        self.identity_filter = tf.constant(identity, tf.float32)
        self.reshape_layer = tf.keras.layers.Reshape((-1, window_width * num_channels))

    def call(self, x):
        x = tf.nn.conv1d(
            input=x, filters=self.identity_filter, stride=1, padding="SAME"
        )
        return self.reshape_layer(x)

    def reshape(self, x):
        return self.reshape_layer(x)


class Dense(tf.keras.layers.Dense):
    def __init__(self, units, dropout_rate, layer_norm, **kwargs):
        super().__init__(units, **kwargs)

        self.dropout = tf.keras.layers.Dropout(dropout_rate) if dropout_rate != 0. else tf.keras.layers.Lambda(lambda x: x)
        self.layer_norm = tf.keras.layers.LayerNormalization() if layer_norm else tf.keras.layers.Lambda(lambda x: x)

    def call(self, x):
        return self.dropout(self.layer_norm(super().call(x)))


class Model(tf.keras.Model):
    def __init__(self, dropout_rates=None, overlap=True, unroll=False, masking=True):
        super().__init__(name='DeepSpeech_v{}'.format(int(file_relative_read('GRAPH_VERSION').strip())))

        if dropout_rates is None:
            dropout_rates = defaultdict(lambda: 0.)

        self.overlap_layer = CreateOverlappingWindows()
        self.overlap_op = self.overlap_layer if overlap else self.overlap_layer.reshape

        self.mask = tf.keras.layers.Masking() if masking else tf.keras.layers.Lambda(lambda x: x)

        def clipped_relu(x):
            return tf.minimum(tf.nn.relu(x), FLAGS.relu_clip)

        self.dense1 = Dense(Config.n_hidden_1, dropout_rate=dropout_rates[0], activation=clipped_relu, layer_norm=FLAGS.layer_norm, name='dense1')
        self.dense2 = Dense(Config.n_hidden_2, dropout_rate=dropout_rates[1], activation=clipped_relu, layer_norm=FLAGS.layer_norm, name='dense2')
        self.dense3 = Dense(Config.n_hidden_3, dropout_rate=dropout_rates[2], activation=clipped_relu, layer_norm=FLAGS.layer_norm, name='dense3')

        self.lstm = tf.keras.layers.LSTM(Config.n_cell_dim, return_sequences=True, return_state=True, unroll=unroll)

        self.dense5 = Dense(Config.n_hidden_5, dropout_rate=dropout_rates[4], activation=clipped_relu, layer_norm=FLAGS.layer_norm, name='dense5')
        self.dense6 = Dense(Config.n_hidden_6, dropout_rate=0., activation='linear', layer_norm=False, name='dense6')

    def call(self, batch_x, previous_state=None):
        layers = {}

        # Input shape: [batch_size, n_steps (dynamic), n_input + 2*n_input*n_context]
        batch_size = tf.shape(input=batch_x)[0]

        # Create overlapping feature windows if needed
        batch_x = self.overlap_op(batch_x)

        # Apply sequence masking
        batch_x = self.mask(batch_x)

        # Apply three dense layers with clipped RELU activation and dropout.
        layers['layer_1'] = layer_1 = self.dense1(batch_x)
        layers['layer_2'] = layer_2 = self.dense2(layer_1)
        layers['layer_3'] = layer_3 = self.dense3(layer_2)

        # RNN layer
        output, output_state_h, output_state_c = self.lstm(layer_3, previous_state)
        layers['rnn_output'] = output
        layers['rnn_output_state_h'] = output_state_h
        layers['rnn_output_state_c'] = output_state_c

        # Now we feed `output` to the fifth dense layer with clipped RELU activation and dropout.
        layers['layer_5'] = layer_5 = self.dense5(output)

        # Now we apply a final linear layer creating `n_classes` dimensional vectors, the logits.
        layers['layer_6'] = layer_6 = self.dense6(layer_5)

        # Finally we reshape layer_6 from a tensor of shape [batch_size*n_steps, n_hidden_6]
        # to the slightly more useful shape [n_steps, batch_size, n_hidden_6].
        # And transpose to time major.
        output = tf.reshape(layer_6, [batch_size, -1, Config.n_hidden_6])
        output = tf.transpose(output, [1, 0, 2], name='raw_logits')
        layers['raw_logits'] = output

        # Output shape: [batch_size, n_steps, n_hidden_6]
        return output, layers



# Accuracy and Loss
# =================

# In accord with 'Deep Speech: Scaling up end-to-end speech recognition'
# (http://arxiv.org/abs/1412.5567),
# the loss function used by our network should be the CTC loss function
# (http://www.cs.toronto.edu/~graves/preprint.pdf).
# Conveniently, this loss function is implemented in TensorFlow.
# Thus, we can simply make use of this implementation to define our loss.

def calculate_mean_edit_distance_and_loss(model, iterator):
    r'''
    This routine beam search decodes a mini-batch and calculates the loss and mean edit distance.
    Next to total and average loss it returns the mean edit distance,
    the decoded result and the batch's original Y.
    '''
    # Obtain the next batch of data
    batch_filenames, (batch_x, batch_seq_len), batch_y = iterator.get_next()

    # Calculate the logits of the batch
    logits, _ = model(batch_x)

    # Compute the CTC loss using TensorFlow's `ctc_loss`
    total_loss = tfv1.nn.ctc_loss(labels=batch_y, inputs=logits, sequence_length=batch_seq_len)

    # Check if any files lead to non finite loss
    non_finite_files = tf.gather(batch_filenames, tfv1.where(~tf.math.is_finite(total_loss)))

    # Calculate the average loss across the batch
    avg_loss = tf.reduce_mean(input_tensor=total_loss)

    # Finally we return the average loss
    return avg_loss, non_finite_files


# Adam Optimization
# =================

# In contrast to 'Deep Speech: Scaling up end-to-end speech recognition'
# (http://arxiv.org/abs/1412.5567),
# in which 'Nesterov's Accelerated Gradient Descent'
# (www.cs.toronto.edu/~fritz/absps/momentum.pdf) was used,
# we will use the Adam method for optimization (http://arxiv.org/abs/1412.6980),
# because, generally, it requires less fine-tuning.
def create_optimizer(learning_rate_var):
    optimizer = tfv1.train.AdamOptimizer(learning_rate=learning_rate_var,
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

def get_tower_results(model, iterator, optimizer):
    r'''
    With this preliminary step out of the way, we can for each GPU introduce a
    tower for which's batch we calculate and return the optimization gradients
    and the average loss across towers.
    '''
    # To calculate the mean of the losses
    tower_avg_losses = []

    # Tower gradients to return
    tower_gradients = []

    # Aggregate any non finite files in the batches
    tower_non_finite_files = []

    with tfv1.variable_scope(tfv1.get_variable_scope()):
        # Loop over available_devices
        for i in range(len(Config.available_devices)):
            # Execute operations of tower i on device i
            device = Config.available_devices[i]
            with tf.device(device):
                # Calculate the avg_loss and mean_edit_distance and retrieve the decoded
                # batch along with the original batch's labels (Y) of this tower
                avg_loss, non_finite_files = calculate_mean_edit_distance_and_loss(model, iterator)

                # Retain tower's avg losses
                tower_avg_losses.append(avg_loss)

                # Compute gradients for model parameters using tower's mini-batch
                gradients = optimizer.compute_gradients(avg_loss)

                # Retain tower's gradients
                tower_gradients.append(gradients)

                tower_non_finite_files.append(non_finite_files)

    avg_loss_across_towers = tf.reduce_mean(input_tensor=tower_avg_losses, axis=0)
    tfv1.summary.scalar(name='step_loss', tensor=avg_loss_across_towers, collections=['step_summaries'])

    all_non_finite_files = tf.concat(tower_non_finite_files, axis=0)

    # Return gradients and the average loss
    return tower_gradients, avg_loss_across_towers, all_non_finite_files


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
            grad = tf.reduce_mean(input_tensor=grad, axis=0)

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
    mean = tf.reduce_mean(input_tensor=variable)
    tfv1.summary.scalar(name='%s/mean'   % name, tensor=mean)
    tfv1.summary.scalar(name='%s/sttdev' % name, tensor=tf.sqrt(tf.reduce_mean(input_tensor=tf.square(variable - mean))))
    tfv1.summary.scalar(name='%s/max'    % name, tensor=tf.reduce_max(input_tensor=variable))
    tfv1.summary.scalar(name='%s/min'    % name, tensor=tf.reduce_min(input_tensor=variable))
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


def train():
    exception_box = ExceptionBox()

    with tfv1.Session(config=Config.session_config) as session:
        log_debug('Session opened.')
        tfv1.keras.backend.set_session(session)

        # Create training and validation datasets
        train_set = create_dataset(FLAGS.train_files.split(','),
                                   batch_size=FLAGS.train_batch_size,
                                   epochs=FLAGS.epochs,
                                   augmentations=Config.augmentations,
                                   cache_path=FLAGS.feature_cache,
                                   train_phase=True,
                                   exception_box=exception_box,
                                   process_ahead=len(Config.available_devices) * FLAGS.train_batch_size * 2,
                                   reverse=FLAGS.reverse_train,
                                   limit=FLAGS.limit_train,
                                   buffering=FLAGS.read_buffer)

        iterator = tfv1.data.Iterator.from_structure(tfv1.data.get_output_types(train_set),
                                                     tfv1.data.get_output_shapes(train_set),
                                                     output_classes=tfv1.data.get_output_classes(train_set))

        # Make initialization ops for switching between the two sets
        train_init_op = iterator.make_initializer(train_set)

        if FLAGS.dev_files:
            dev_sources = FLAGS.dev_files.split(',')
            dev_sets = [create_dataset([source],
                                       batch_size=FLAGS.dev_batch_size,
                                       train_phase=False,
                                       exception_box=exception_box,
                                       process_ahead=len(Config.available_devices) * FLAGS.dev_batch_size * 2,
                                       reverse=FLAGS.reverse_dev,
                                       limit=FLAGS.limit_dev,
                                       buffering=FLAGS.read_buffer) for source in dev_sources]
            dev_init_ops = [iterator.make_initializer(dev_set) for dev_set in dev_sets]

        if FLAGS.metrics_files:
            metrics_sources = FLAGS.metrics_files.split(',')
            metrics_sets = [create_dataset([source],
                                           batch_size=FLAGS.dev_batch_size,
                                           train_phase=False,
                                           exception_box=exception_box,
                                           process_ahead=len(Config.available_devices) * FLAGS.dev_batch_size * 2,
                                           reverse=FLAGS.reverse_dev,
                                           limit=FLAGS.limit_dev,
                                           buffering=FLAGS.read_buffer) for source in metrics_sources]
            metrics_init_ops = [iterator.make_initializer(metrics_set) for metrics_set in metrics_sets]

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
        learning_rate_var = tfv1.get_variable('learning_rate', initializer=FLAGS.learning_rate, trainable=False)
        reduce_learning_rate_op = learning_rate_var.assign(tf.multiply(learning_rate_var, FLAGS.plateau_reduction))
        optimizer = create_optimizer(learning_rate_var)

        # Enable mixed precision training
        if FLAGS.automatic_mixed_precision:
            log_info('Enabling automatic mixed precision training.')
            optimizer = tfv1.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)

        model = Model(dropout_rates)

        gradients, loss, non_finite_files = get_tower_results(model, iterator, optimizer)

        # Average tower gradients across GPUs
        avg_tower_gradients = average_gradients(gradients)
        log_grads_and_vars(avg_tower_gradients)

        # global_step is incremented by the optimizer
        global_step = tfv1.train.get_or_create_global_step()
        apply_gradient_op = optimizer.apply_gradients(avg_tower_gradients, global_step)

        # Summaries
        step_summaries_op = tfv1.summary.merge_all('step_summaries')
        step_summary_writers = {
            'train': tfv1.summary.FileWriter(os.path.join(FLAGS.summary_dir, 'train'), max_queue=120),
            'dev': tfv1.summary.FileWriter(os.path.join(FLAGS.summary_dir, 'dev'), max_queue=120),
            'metrics': tfv1.summary.FileWriter(os.path.join(FLAGS.summary_dir, 'metrics'), max_queue=120),
        }

        human_readable_set_names = {
            'train': 'Training',
            'dev': 'Validation',
            'metrics': 'Metrics',
        }

        # Checkpointing
        checkpoint_saver = tfv1.train.Saver(max_to_keep=FLAGS.max_to_keep)
        checkpoint_path = os.path.join(FLAGS.save_checkpoint_dir, 'train')

        best_dev_saver = tfv1.train.Saver(max_to_keep=1)
        best_dev_path = os.path.join(FLAGS.save_checkpoint_dir, 'best_dev')

        # Save flags next to checkpoints
        if not is_remote_path(FLAGS.save_checkpoint_dir):
            os.makedirs(FLAGS.save_checkpoint_dir, exist_ok=True)
        flags_file = os.path.join(FLAGS.save_checkpoint_dir, 'flags.txt')
        with open_remote(flags_file, 'w') as fout:
            fout.write(FLAGS.flags_into_string())

        # Prevent further graph changes
        tfv1.get_default_graph().finalize()

        # Load checkpoint or initialize variables
        load_or_init_graph_for_training(session)

        def run_set(set_name, epoch, init_op, dataset=None):
            is_train = set_name == 'train'
            train_op = apply_gradient_op if is_train else []
            feed_dict = dropout_feed_dict if is_train else no_dropout_feed_dict

            total_loss = 0.0
            step_count = 0

            step_summary_writer = step_summary_writers.get(set_name)
            checkpoint_time = time.time()

            if is_train and FLAGS.cache_for_epochs > 0 and FLAGS.feature_cache:
                feature_cache_index = FLAGS.feature_cache + '.index'
                if epoch % FLAGS.cache_for_epochs == 0 and os.path.isfile(feature_cache_index):
                    log_info('Invalidating feature cache')
                    remove_remote(feature_cache_index)  # this will let TF also overwrite the related cache data files

            # Setup progress bar
            class LossWidget(progressbar.widgets.FormatLabel):
                def __init__(self):
                    progressbar.widgets.FormatLabel.__init__(self, format='Loss: %(mean_loss)f')

                def __call__(self, progress, data, **kwargs):
                    data['mean_loss'] = total_loss / step_count if step_count else 0.0
                    return progressbar.widgets.FormatLabel.__call__(self, progress, data, **kwargs)

            prefix = 'Epoch {} | {:>10}'.format(epoch, human_readable_set_names[set_name])
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
                    _, current_step, batch_loss, problem_files, step_summary = \
                        session.run([train_op, global_step, loss, non_finite_files, step_summaries_op],
                                    feed_dict=feed_dict)
                    exception_box.raise_if_set()
                except tf.errors.OutOfRangeError:
                    exception_box.raise_if_set()
                    break

                if problem_files.size > 0:
                    problem_files = [f.decode('utf8') for f in problem_files[..., 0]]
                    log_error('The following files caused an infinite (or NaN) '
                              'loss: {}'.format(','.join(problem_files)))

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
        epochs_without_improvement = 0
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
                    for source, init_op in zip(dev_sources, dev_init_ops):
                        log_progress('Validating epoch %d on %s...' % (epoch, source))
                        set_loss, steps = run_set('dev', epoch, init_op, dataset=source)
                        dev_loss += set_loss * steps
                        total_steps += steps
                        log_progress('Finished validating epoch %d on %s - loss: %f' % (epoch, source, set_loss))

                    dev_loss = dev_loss / total_steps
                    dev_losses.append(dev_loss)

                    # Count epochs without an improvement for early stopping and reduction of learning rate on a plateau
                    # the improvement has to be greater than FLAGS.es_min_delta
                    if dev_loss > best_dev_loss - FLAGS.es_min_delta:
                        epochs_without_improvement += 1
                    else:
                        epochs_without_improvement = 0

                    # Save new best model
                    if dev_loss < best_dev_loss:
                        best_dev_loss = dev_loss
                        save_path = best_dev_saver.save(session, best_dev_path, global_step=global_step, latest_filename='best_dev_checkpoint')
                        log_info("Saved new best validating model with loss %f to: %s" % (best_dev_loss, save_path))

                    # Early stopping
                    if FLAGS.early_stop and epochs_without_improvement == FLAGS.es_epochs:
                        log_info('Early stop triggered as the loss did not improve the last {} epochs'.format(
                            epochs_without_improvement))
                        break

                    # Reduce learning rate on plateau
                    # If the learning rate was reduced and there is still no improvement
                    # wait FLAGS.plateau_epochs before the learning rate is reduced again
                    if (
                        FLAGS.reduce_lr_on_plateau
                        and epochs_without_improvement > 0
                        and epochs_without_improvement % FLAGS.plateau_epochs == 0
                    ):
                        # Reload checkpoint that we use the best_dev weights again
                        reload_best_checkpoint(session)

                        # Reduce learning rate
                        session.run(reduce_learning_rate_op)
                        current_learning_rate = learning_rate_var.eval()
                        log_info('Encountered a plateau, reducing learning rate to {}'.format(
                            current_learning_rate))

                        # Overwrite best checkpoint with new learning rate value
                        save_path = best_dev_saver.save(session, best_dev_path, global_step=global_step, latest_filename='best_dev_checkpoint')
                        log_info("Saved best validating model with reduced learning rate to: %s" % (save_path))

                if FLAGS.metrics_files:
                    # Read only metrics, not affecting best validation loss tracking
                    for source, init_op in zip(metrics_sources, metrics_init_ops):
                        log_progress('Metrics for epoch %d on %s...' % (epoch, source))
                        set_loss, _ = run_set('metrics', epoch, init_op, dataset=source)
                        log_progress('Metrics for epoch %d on %s - loss: %f' % (epoch, source, set_loss))

                print('-' * 80)


        except KeyboardInterrupt:
            pass
        log_info('FINISHED optimization in {}'.format(datetime.utcnow() - train_start_time))
    log_debug('Session closed.')


def test():
    samples = evaluate(FLAGS.test_files.split(','), Model)
    if FLAGS.test_output_file:
        save_samples_json(samples, FLAGS.test_output_file)


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

    if batch_size <= 0:
        # no state management since n_step is expected to be dynamic too (see below)
        previous_state = None
    else:
        previous_state_c = tfv1.placeholder(tf.float32, [batch_size, Config.n_cell_dim], name='previous_state_c')
        previous_state_h = tfv1.placeholder(tf.float32, [batch_size, Config.n_cell_dim], name='previous_state_h')

        previous_state = [previous_state_h, previous_state_c]

    model = Model(overlap=False, unroll=n_steps > 0)
    logits, layers = model(input_tensor,
                           previous_state)

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
            },
            {
                'outputs': probs,
            },
            layers
        )

    new_state_c = layers['rnn_output_state_c']
    new_state_h = layers['rnn_output_state_h']
    new_state_c = tf.identity(new_state_c, name='new_state_c')
    new_state_h = tf.identity(new_state_h, name='new_state_h')

    inputs = {
        'input': input_tensor,
        'previous_state_c': previous_state_c,
        'previous_state_h': previous_state_h,
        'input_samples': input_samples,
    }

    outputs = {
        'outputs': probs,
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

    with tfv1.Session() as session:
        inputs, outputs, _ = create_inference_graph(batch_size=FLAGS.export_batch_size, n_steps=FLAGS.n_steps, tflite=FLAGS.export_tflite)

        graph_version = int(file_relative_read('GRAPH_VERSION').strip())
        assert graph_version > 0

        outputs['metadata_version'] = tf.constant([graph_version], name='metadata_version')
        outputs['metadata_sample_rate'] = tf.constant([FLAGS.audio_sample_rate], name='metadata_sample_rate')
        outputs['metadata_feature_win_len'] = tf.constant([FLAGS.feature_win_len], name='metadata_feature_win_len')
        outputs['metadata_feature_win_step'] = tf.constant([FLAGS.feature_win_step], name='metadata_feature_win_step')
        outputs['metadata_beam_width'] = tf.constant([FLAGS.export_beam_width], name='metadata_beam_width')
        outputs['metadata_alphabet'] = tf.constant([Config.alphabet.Serialize()], name='metadata_alphabet')

        if FLAGS.export_language:
            outputs['metadata_language'] = tf.constant([FLAGS.export_language.encode('utf-8')], name='metadata_language')

        # Prevent further graph changes
        tfv1.get_default_graph().finalize()

        output_names_tensors = [tensor.op.name for tensor in outputs.values() if isinstance(tensor, tf.Tensor)]
        output_names_ops = [op.name for op in outputs.values() if isinstance(op, tf.Operation)]
        output_names = output_names_tensors + output_names_ops

        # Restore variables from checkpoint
        load_graph_for_evaluation(session)

        output_filename = FLAGS.export_file_name + '.pb'
        if FLAGS.remove_export:
            if isdir_remote(FLAGS.export_dir):
                log_info('Removing old export')
                rmtree_remote(FLAGS.export_dir)

        output_graph_path = os.path.join(FLAGS.export_dir, output_filename)

        if not is_remote_path(FLAGS.export_dir) and not os.path.isdir(FLAGS.export_dir):
            os.makedirs(FLAGS.export_dir)

        frozen_graph = tfv1.graph_util.convert_variables_to_constants(
            sess=session,
            input_graph_def=tfv1.get_default_graph().as_graph_def(),
            output_node_names=output_names)

        frozen_graph = tfv1.graph_util.extract_sub_graph(
            graph_def=frozen_graph,
            dest_nodes=output_names)

        if not FLAGS.export_tflite:
            with open_remote(output_graph_path, 'wb') as fout:
                fout.write(frozen_graph.SerializeToString())
        else:
            output_tflite_path = os.path.join(FLAGS.export_dir, output_filename.replace('.pb', '.tflite'))

            converter = tfv1.lite.TFLiteConverter.from_session(session, input_tensors=list(inputs.values()), output_tensors=list(outputs.values()))
            # New converter breaks our export
            converter.experimental_new_converter = False
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            # AudioSpectrogram and Mfcc ops are custom but have built-in kernels in TFLite
            converter.allow_custom_ops = True
            tflite_model = converter.convert()

            with open_remote(output_tflite_path, 'wb') as fout:
                fout.write(tflite_model)

        log_info('Models exported at %s' % (FLAGS.export_dir))

    metadata_fname = os.path.join(FLAGS.export_dir, '{}_{}_{}.md'.format(
        FLAGS.export_author_id,
        FLAGS.export_model_name,
        FLAGS.export_model_version))

    model_runtime = 'tflite' if FLAGS.export_tflite else 'tensorflow'
    with open_remote(metadata_fname, 'w') as f:
        f.write('---\n')
        f.write('author: {}\n'.format(FLAGS.export_author_id))
        f.write('model_name: {}\n'.format(FLAGS.export_model_name))
        f.write('model_version: {}\n'.format(FLAGS.export_model_version))
        f.write('contact_info: {}\n'.format(FLAGS.export_contact_info))
        f.write('license: {}\n'.format(FLAGS.export_license))
        f.write('language: {}\n'.format(FLAGS.export_language))
        f.write('runtime: {}\n'.format(model_runtime))
        f.write('min_ds_version: {}\n'.format(FLAGS.export_min_ds_version))
        f.write('max_ds_version: {}\n'.format(FLAGS.export_max_ds_version))
        f.write('acoustic_model_url: <replace this with a publicly available URL of the acoustic model>\n')
        f.write('scorer_url: <replace this with a publicly available URL of the scorer, if present>\n')
        f.write('---\n')
        f.write('{}\n'.format(FLAGS.export_description))

    log_info('Model metadata file saved to {}. Before submitting the exported model for publishing make sure all information in the metadata file is correct, and complete the URL fields.'.format(metadata_fname))


def package_zip():
    # --export_dir path/to/export/LANG_CODE/ => path/to/export/LANG_CODE.zip
    export_dir = os.path.join(os.path.abspath(FLAGS.export_dir), '') # Force ending '/'
    if is_remote_path(export_dir):
        log_error("Cannot package remote path zip %s. Please do this manually." % export_dir)
        return

    zip_filename = os.path.dirname(export_dir)
    
    shutil.copy(FLAGS.scorer_path, export_dir)

    archive = shutil.make_archive(zip_filename, 'zip', export_dir)
    log_info('Exported packaged model {}'.format(archive))


def do_single_file_inference(input_file_path):
    with tfv1.Session(config=Config.session_config) as session:
        chunk_size = 16
        inputs, outputs, _ = create_inference_graph(batch_size=1, n_steps=chunk_size)

        # Restore variables from training checkpoint
        load_graph_for_evaluation(session)

        features, features_len = audiofile_to_features(input_file_path)
        previous_state_c = np.zeros([1, Config.n_cell_dim])
        previous_state_h = np.zeros([1, Config.n_cell_dim])

        # Add batch dimension
        features = tf.expand_dims(features, 0)

        # Evaluate
        features = create_overlapping_windows(features).eval(session=session)

        # Run through acoustic models in chunks to simulate behavior of native client
        probs = []
        for i in range(0, features.shape[1], chunk_size):
            input_chunk = features[:, i:i + chunk_size, ...]
            chunk_len = input_chunk.shape[1]
            input_chunk = np.pad(input_chunk, ((0, 0), (0, chunk_size-chunk_len), (0, 0), (0, 0)))
            probs_chunk, new_state_c, new_state_h = session.run(
                (outputs['outputs'], outputs['new_state_c'], outputs['new_state_h']),
                feed_dict={
                    inputs['input']: input_chunk,
                    inputs['previous_state_c']: previous_state_c,
                    inputs['previous_state_h']: previous_state_h,
                })
            probs.append(probs_chunk[:chunk_len,...])
            previous_state_c = new_state_c
            previous_state_h = new_state_h

        probs = np.squeeze(np.concatenate(probs))

        if FLAGS.scorer_path:
            scorer = Scorer(FLAGS.lm_alpha, FLAGS.lm_beta,
                            FLAGS.scorer_path, Config.alphabet)
        else:
            scorer = None
        decoded = ctc_beam_search_decoder(probs, Config.alphabet, FLAGS.beam_width,
                                          scorer=scorer, cutoff_prob=FLAGS.cutoff_prob,
                                          cutoff_top_n=FLAGS.cutoff_top_n)
        # Print highest probability result
        print(decoded[0][1])


def early_training_checks():
    # Check for proper scorer early
    if FLAGS.scorer_path:
        scorer = Scorer(FLAGS.lm_alpha, FLAGS.lm_beta,
                        FLAGS.scorer_path, Config.alphabet)
        del scorer

    if FLAGS.train_files and FLAGS.test_files and FLAGS.load_checkpoint_dir != FLAGS.save_checkpoint_dir:
        log_warn('WARNING: You specified different values for --load_checkpoint_dir '
                 'and --save_checkpoint_dir, but you are running training and testing '
                 'in a single invocation. The testing step will respect --load_checkpoint_dir, '
                 'and thus WILL NOT TEST THE CHECKPOINT CREATED BY THE TRAINING STEP. '
                 'Train and test in two separate invocations, specifying the correct '
                 '--load_checkpoint_dir in both cases, or use the same location '
                 'for loading and saving.')


def main(_):
    initialize_globals()
    early_training_checks()

    if FLAGS.train_files:
        tfv1.reset_default_graph()
        tfv1.set_random_seed(FLAGS.random_seed)
        train()

    if FLAGS.test_files:
        tfv1.reset_default_graph()
        test()

    if FLAGS.export_dir and not FLAGS.export_zip:
        tfv1.reset_default_graph()
        export()

    if FLAGS.export_zip:
        tfv1.reset_default_graph()
        FLAGS.export_tflite = True

        if listdir_remote(FLAGS.export_dir):
            log_error('Directory {} is not empty, please fix this.'.format(FLAGS.export_dir))
            sys.exit(1)

        export()
        package_zip()

    if FLAGS.one_shot_infer:
        tfv1.reset_default_graph()
        do_single_file_inference(FLAGS.one_shot_infer)


def run_script():
    create_flags()
    absl.app.run(main)

if __name__ == '__main__':
    run_script()
