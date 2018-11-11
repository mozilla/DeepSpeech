from __future__ import absolute_import, division, print_function

import os
import tensorflow as tf

from attrdict import AttrDict
from six.moves import zip, range, filter
from util.gpu import get_available_gpus
from util.logging import log_error
from util.text import Alphabet
from xdg import BaseDirectory as xdg

class GlobalConfig:
    _config = None

    def __getattr__(self, name):
        if not GlobalConfig._config:
            raise RuntimeError("Global configuration not yet initialized.")
        if not hasattr(GlobalConfig._config, name):
            raise RuntimeError("Configuration option {} not found in config.".format(name))
        return GlobalConfig._config[name]


C = GlobalConfig()

def initialize_globals():
    c = AttrDict()

    # ps and worker hosts required for p2p cluster setup
    FLAGS.ps_hosts = list(filter(len, FLAGS.ps_hosts.split(',')))
    FLAGS.worker_hosts = list(filter(len, FLAGS.worker_hosts.split(',')))

    # The absolute number of computing nodes - regardless of cluster or single mode
    c.num_workers = max(1, len(FLAGS.worker_hosts))

    # Create a cluster from the parameter server and worker hosts.
    c.cluster = tf.train.ClusterSpec({'ps': FLAGS.ps_hosts, 'worker': FLAGS.worker_hosts})

    # If replica numbers are negative, we multiply their absolute values with the number of workers
    if FLAGS.replicas < 0:
        FLAGS.replicas = c.num_workers * -FLAGS.replicas
    if FLAGS.replicas_to_agg < 0:
        FLAGS.replicas_to_agg = c.num_workers * -FLAGS.replicas_to_agg

    # The device path base for this node
    c.worker_device = '/job:%s/task:%d' % (FLAGS.job_name, FLAGS.task_index)

    # This node's CPU device
    c.cpu_device = c.worker_device + '/cpu:0'

    # This node's available GPU devices
    c.available_devices = [c.worker_device + gpu for gpu in get_available_gpus()]

    # If there is no GPU available, we fall back to CPU based operation
    if 0 == len(c.available_devices):
        c.available_devices = [c.cpu_device]

    # Set default dropout rates
    if FLAGS.dropout_rate2 < 0:
        FLAGS.dropout_rate2 = FLAGS.dropout_rate
    if FLAGS.dropout_rate3 < 0:
        FLAGS.dropout_rate3 = FLAGS.dropout_rate
    if FLAGS.dropout_rate6 < 0:
        FLAGS.dropout_rate6 = FLAGS.dropout_rate

    c.no_dropout = [ 0.0 ] * 6

    # Set default checkpoint dir
    if len(FLAGS.checkpoint_dir) == 0:
        FLAGS.checkpoint_dir = xdg.save_data_path(os.path.join('deepspeech','checkpoints'))

    # Set default summary dir
    if len(FLAGS.summary_dir) == 0:
        FLAGS.summary_dir = xdg.save_data_path(os.path.join('deepspeech','summaries'))

    # Standard session configuration that'll be used for all new sessions.
    c.session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=FLAGS.log_placement,
                                      inter_op_parallelism_threads=FLAGS.inter_op_parallelism_threads,
                                      intra_op_parallelism_threads=FLAGS.intra_op_parallelism_threads)

    c.alphabet = Alphabet(os.path.abspath(FLAGS.alphabet_config_path))

    # Geometric Constants
    # ===================

    # For an explanation of the meaning of the geometric constants, please refer to
    # doc/Geometry.md

    # Number of MFCC features
    c.n_input = 26 # TODO: Determine this programatically from the sample rate

    # The number of frames in the context
    c.n_context = 9 # TODO: Determine the optimal value using a validation data set

    # Number of units in hidden layers
    c.n_hidden = FLAGS.n_hidden

    c.n_hidden_1 = c.n_hidden

    c.n_hidden_2 = c.n_hidden

    c.n_hidden_5 = c.n_hidden

    # LSTM cell state dimension
    c.n_cell_dim = c.n_hidden

    # The number of units in the third layer, which feeds in to the LSTM
    c.n_hidden_3 = c.n_cell_dim

    # The number of characters in the target language plus one
    c.n_character = c.alphabet.size() + 1 # +1 for CTC blank label

    # The number of units in the sixth layer
    c.n_hidden_6 = c.n_character

    # Queues that are used to gracefully stop parameter servers.
    # Each queue stands for one ps. A finishing worker sends a token to each queue before joining/quitting.
    # Each ps will dequeue as many tokens as there are workers before joining/quitting.
    # This ensures parameter servers won't quit, if still required by at least one worker and
    # also won't wait forever (like with a standard `server.join()`).
    c.done_queues = []
    for i, ps in enumerate(FLAGS.ps_hosts):
        # Queues are hosted by their respective owners
        with tf.device('/job:ps/task:%d' % i):
            c.done_queues.append(tf.FIFOQueue(1, tf.int32, shared_name=('queue%i' % i)))

    # Placeholder to pass in the worker's index as token
    c.token_placeholder = tf.placeholder(tf.int32)

    # Enqueue operations for each parameter server
    c.done_enqueues = [queue.enqueue(token_placeholder) for queue in c.done_queues]

    # Dequeue operations for each parameter server
    c.done_dequeues = [queue.dequeue() for queue in c.done_queues]

    if len(FLAGS.one_shot_infer) > 0:
        FLAGS.train = False
        FLAGS.test = False
        FLAGS.export_dir = ''
        if not os.path.exists(FLAGS.one_shot_infer):
            log_error('Path specified in --one_shot_infer is not a valid file.')
            exit(1)

    # Determine, if we are the chief worker
    c.is_chief = len(FLAGS.worker_hosts) == 0 or (FLAGS.task_index == 0 and FLAGS.job_name == 'worker')

    GlobalConfig._config = c
