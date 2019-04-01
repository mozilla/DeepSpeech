from __future__ import absolute_import, division, print_function

import os
import tensorflow as tf

from attrdict import AttrDict
from util.flags import FLAGS
from util.gpu import get_available_gpus
from util.logging import log_error
from util.text import Alphabet
from xdg import BaseDirectory as xdg

class ConfigSingleton:
    _config = None

    def __getattr__(self, name):
        if not ConfigSingleton._config:
            raise RuntimeError("Global configuration not yet initialized.")
        if not hasattr(ConfigSingleton._config, name):
            raise RuntimeError("Configuration option {} not found in config.".format(name))
        return ConfigSingleton._config[name]


Config = ConfigSingleton()

def initialize_globals():
    c = AttrDict()

    # CPU device
    c.cpu_device = '/cpu:0'

    # Available GPU devices
    c.available_devices = get_available_gpus()

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

    # Set default checkpoint dir
    if len(FLAGS.checkpoint_dir) == 0:
        FLAGS.checkpoint_dir = xdg.save_data_path(os.path.join('deepspeech','checkpoints'))

    if FLAGS.load not in ['last', 'best', 'init', 'auto']:
        FLAGS.load = 'auto'

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
    c.n_input = 26 # TODO: Determine this programmatically from the sample rate

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

    # Units in the sixth layer = number of characters in the target language plus one
    c.n_hidden_6 = c.alphabet.size() + 1 # +1 for CTC blank label

    if len(FLAGS.one_shot_infer) > 0:
        FLAGS.train = False
        FLAGS.test = False
        FLAGS.export_dir = ''
        if not os.path.exists(FLAGS.one_shot_infer):
            log_error('Path specified in --one_shot_infer is not a valid file.')
            exit(1)

    ConfigSingleton._config = c
