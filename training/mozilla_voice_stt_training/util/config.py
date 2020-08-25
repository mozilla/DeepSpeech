from __future__ import absolute_import, division, print_function

import os
import sys
import tensorflow.compat.v1 as tfv1

from attrdict import AttrDict
from xdg import BaseDirectory as xdg
from ds_ctcdecoder import Alphabet, UTF8Alphabet

from .flags import FLAGS
from .gpu import get_available_gpus
from .logging import log_error, log_warn
from .helpers import parse_file_size
from .augmentations import parse_augmentations


class ConfigSingleton:
    _config = None

    def __getattr__(self, name):
        if not ConfigSingleton._config:
            raise RuntimeError("Global configuration not yet initialized.")
        if not hasattr(ConfigSingleton._config, name):
            raise RuntimeError("Configuration option {} not found in config.".format(name))
        return ConfigSingleton._config[name]


Config = ConfigSingleton() # pylint: disable=invalid-name

def initialize_globals():
    c = AttrDict()

    # Augmentations
    c.augmentations = parse_augmentations(FLAGS.augment)
    if len(c.augmentations) > 0 and FLAGS.feature_cache and FLAGS.cache_for_epochs == 0:
        log_warn('Due to current feature-cache settings the exact same sample augmentations of the first '
                 'epoch will be repeated on all following epochs. This could lead to unintended over-fitting. '
                 'You could use --cache_for_epochs <n_epochs> to invalidate the cache after a given number of epochs.')

    # Caching
    if FLAGS.cache_for_epochs == 1:
        log_warn('--cache_for_epochs == 1 is (re-)creating the feature cache on every epoch but will never use it.')

    # Read-buffer
    FLAGS.read_buffer = parse_file_size(FLAGS.read_buffer)

    # Set default dropout rates
    if FLAGS.dropout_rate2 < 0:
        FLAGS.dropout_rate2 = FLAGS.dropout_rate
    if FLAGS.dropout_rate3 < 0:
        FLAGS.dropout_rate3 = FLAGS.dropout_rate
    if FLAGS.dropout_rate6 < 0:
        FLAGS.dropout_rate6 = FLAGS.dropout_rate

    # Set default checkpoint dir
    if not FLAGS.checkpoint_dir:
        FLAGS.checkpoint_dir = xdg.save_data_path(os.path.join('deepspeech', 'checkpoints'))

    if FLAGS.load_train not in ['last', 'best', 'init', 'auto']:
        FLAGS.load_train = 'auto'

    if FLAGS.load_evaluate not in ['last', 'best', 'auto']:
        FLAGS.load_evaluate = 'auto'

    # Set default summary dir
    if not FLAGS.summary_dir:
        FLAGS.summary_dir = xdg.save_data_path(os.path.join('deepspeech', 'summaries'))

    # Standard session configuration that'll be used for all new sessions.
    c.session_config = tfv1.ConfigProto(allow_soft_placement=True, log_device_placement=FLAGS.log_placement,
                                        inter_op_parallelism_threads=FLAGS.inter_op_parallelism_threads,
                                        intra_op_parallelism_threads=FLAGS.intra_op_parallelism_threads,
                                        gpu_options=tfv1.GPUOptions(allow_growth=FLAGS.use_allow_growth))

    # CPU device
    c.cpu_device = '/cpu:0'

    # Available GPU devices
    c.available_devices = get_available_gpus(c.session_config)

    # If there is no GPU available, we fall back to CPU based operation
    if not c.available_devices:
        c.available_devices = [c.cpu_device]

    if FLAGS.utf8:
        c.alphabet = UTF8Alphabet()
    else:
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
    c.n_hidden_6 = c.alphabet.GetSize() + 1 # +1 for CTC blank label

    # Size of audio window in samples
    if (FLAGS.feature_win_len * FLAGS.audio_sample_rate) % 1000 != 0:
        log_error('--feature_win_len value ({}) in milliseconds ({}) multiplied '
                  'by --audio_sample_rate value ({}) must be an integer value. Adjust '
                  'your --feature_win_len value or resample your audio accordingly.'
                  ''.format(FLAGS.feature_win_len, FLAGS.feature_win_len / 1000, FLAGS.audio_sample_rate))
        sys.exit(1)

    c.audio_window_samples = FLAGS.audio_sample_rate * (FLAGS.feature_win_len / 1000)

    # Stride for feature computations in samples
    if (FLAGS.feature_win_step * FLAGS.audio_sample_rate) % 1000 != 0:
        log_error('--feature_win_step value ({}) in milliseconds ({}) multiplied '
                  'by --audio_sample_rate value ({}) must be an integer value. Adjust '
                  'your --feature_win_step value or resample your audio accordingly.'
                  ''.format(FLAGS.feature_win_step, FLAGS.feature_win_step / 1000, FLAGS.audio_sample_rate))
        sys.exit(1)

    c.audio_step_samples = FLAGS.audio_sample_rate * (FLAGS.feature_win_step / 1000)

    if FLAGS.one_shot_infer:
        if not os.path.exists(FLAGS.one_shot_infer):
            log_error('Path specified in --one_shot_infer is not a valid file.')
            sys.exit(1)

    if FLAGS.train_cudnn and FLAGS.load_cudnn:
        log_error('Trying to use --train_cudnn, but --load_cudnn '
                  'was also specified. The --load_cudnn flag is only '
                  'needed when converting a CuDNN RNN checkpoint to '
                  'a CPU-capable graph. If your system is capable of '
                  'using CuDNN RNN, you can just specify the CuDNN RNN '
                  'checkpoint normally with --save_checkpoint_dir.')
        sys.exit(1)

    # If separate save and load flags were not specified, default to load and save
    # from the same dir.
    if not FLAGS.save_checkpoint_dir:
        FLAGS.save_checkpoint_dir = FLAGS.checkpoint_dir

    if not FLAGS.load_checkpoint_dir:
        FLAGS.load_checkpoint_dir = FLAGS.checkpoint_dir

    ConfigSingleton._config = c # pylint: disable=protected-access
