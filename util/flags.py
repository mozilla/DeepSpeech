from __future__ import absolute_import, division, print_function

import tensorflow as tf


FLAGS = tf.app.flags.FLAGS


def create_flags():
    # Importer
    # ========

    f = tf.app.flags

    f.DEFINE_string('train_files', '', 'comma separated list of files specifying the dataset used for training. Multiple files will get merged. If empty, training will not be run.')
    f.DEFINE_string('dev_files', '', 'comma separated list of files specifying the dataset used for validation. Multiple files will get merged. If empty, validation will not be run.')
    f.DEFINE_string('test_files', '', 'comma separated list of files specifying the dataset used for testing. Multiple files will get merged. If empty, the model will not be tested.')

    f.DEFINE_string('train_cached_features_path', '', 'comma separated list of files specifying the dataset used for training. multiple files will get merged')

    f.DEFINE_integer('feature_win_len', 32, 'feature extraction audio window length in milliseconds')
    f.DEFINE_integer('feature_win_step', 20, 'feature extraction window step length in milliseconds')
    f.DEFINE_integer('audio_sample_rate', 16000, 'sample rate value expected by model')

    # Global Constants
    # ================

    f.DEFINE_integer('epochs', 75, 'how many epochs (complete runs through the train files) to train for')

    f.DEFINE_float('dropout_rate', 0.05, 'dropout rate for feedforward layers')
    f.DEFINE_float('dropout_rate2', -1.0, 'dropout rate for layer 2 - defaults to dropout_rate')
    f.DEFINE_float('dropout_rate3', -1.0, 'dropout rate for layer 3 - defaults to dropout_rate')
    f.DEFINE_float('dropout_rate4', 0.0, 'dropout rate for layer 4 - defaults to 0.0')
    f.DEFINE_float('dropout_rate5', 0.0, 'dropout rate for layer 5 - defaults to 0.0')
    f.DEFINE_float('dropout_rate6', -1.0, 'dropout rate for layer 6 - defaults to dropout_rate')

    f.DEFINE_float('relu_clip', 20.0, 'ReLU clipping value for non-recurrent layers')

    # Adam optimizer(http://arxiv.org/abs/1412.6980) parameters

    f.DEFINE_float('beta1', 0.9, 'beta 1 parameter of Adam optimizer')
    f.DEFINE_float('beta2', 0.999, 'beta 2 parameter of Adam optimizer')
    f.DEFINE_float('epsilon', 1e-8, 'epsilon parameter of Adam optimizer')
    f.DEFINE_float('learning_rate', 0.001, 'learning rate of Adam optimizer')

    # Batch sizes

    f.DEFINE_integer('train_batch_size', 1, 'number of elements in a training batch')
    f.DEFINE_integer('dev_batch_size', 1, 'number of elements in a validation batch')
    f.DEFINE_integer('test_batch_size', 1, 'number of elements in a test batch')

    f.DEFINE_integer('export_batch_size', 1, 'number of elements per batch on the exported graph')

    # Performance(UNSUPPORTED)
    f.DEFINE_integer('inter_op_parallelism_threads', 0, 'number of inter-op parallelism threads - see tf.ConfigProto for more details')
    f.DEFINE_integer('intra_op_parallelism_threads', 0, 'number of intra-op parallelism threads - see tf.ConfigProto for more details')

    # Sample limits

    f.DEFINE_integer('limit_train', 0, 'maximum number of elements to use from train set - 0 means no limit')
    f.DEFINE_integer('limit_dev', 0, 'maximum number of elements to use from validation set- 0 means no limit')
    f.DEFINE_integer('limit_test', 0, 'maximum number of elements to use from test set- 0 means no limit')

    # Checkpointing

    f.DEFINE_string('checkpoint_dir', '', 'directory in which checkpoints are stored - defaults to directory "deepspeech/checkpoints" within user\'s data home specified by the XDG Base Directory Specification')
    f.DEFINE_integer('checkpoint_secs', 600, 'checkpoint saving interval in seconds')
    f.DEFINE_integer('max_to_keep', 5, 'number of checkpoint files to keep - default value is 5')
    f.DEFINE_string('load', 'auto', '"last" for loading most recent epoch checkpoint, "best" for loading best validated checkpoint, "init" for initializing a fresh model, "auto" for trying the other options in order last > best > init')

    # Exporting

    f.DEFINE_string('export_dir', '', 'directory in which exported models are stored - if omitted, the model won\'t get exported')
    f.DEFINE_integer('export_version', 1, 'version number of the exported model')
    f.DEFINE_boolean('remove_export', False, 'whether to remove old exported models')
    f.DEFINE_boolean('export_tflite', False, 'export a graph ready for TF Lite engine')
    f.DEFINE_boolean('use_seq_length', True, 'have sequence_length in the exported graph(will make tfcompile unhappy)')
    f.DEFINE_integer('n_steps', 16, 'how many timesteps to process at once by the export graph, higher values mean more latency')
    f.DEFINE_string('export_language', '', 'language the model was trained on e.g. "en" or "English". Gets embedded into exported model.')

    # Reporting

    f.DEFINE_integer('log_level', 1, 'log level for console logs - 0: INFO, 1: WARN, 2: ERROR, 3: FATAL')
    f.DEFINE_boolean('show_progressbar', True, 'Show progress for training, validation and testing processes. Log level should be > 0.')

    f.DEFINE_boolean('log_placement', False, 'whether to log device placement of the operators to the console')
    f.DEFINE_integer('report_count', 10, 'number of phrases with lowest WER(best matching) to print out during a WER report')

    f.DEFINE_string('summary_dir', '', 'target directory for TensorBoard summaries - defaults to directory "deepspeech/summaries" within user\'s data home specified by the XDG Base Directory Specification')

    # Geometry

    f.DEFINE_integer('n_hidden', 2048, 'layer width to use when initialising layers')

    # Initialization

    f.DEFINE_integer('random_seed', 4568, 'default random seed that is used to initialize variables')

    # Early Stopping

    f.DEFINE_boolean('early_stop', True, 'enable early stopping mechanism over validation dataset. If validation is not being run, early stopping is disabled.')
    f.DEFINE_integer('es_steps', 4, 'number of validations to consider for early stopping. Loss is not stored in the checkpoint so when checkpoint is revived it starts the loss calculation from start at that point')
    f.DEFINE_float('es_mean_th', 0.5, 'mean threshold for loss to determine the condition if early stopping is required')
    f.DEFINE_float('es_std_th', 0.5, 'standard deviation threshold for loss to determine the condition if early stopping is required')

    # Decoder

    f.DEFINE_string('alphabet_config_path', 'data/alphabet.txt', 'path to the configuration file specifying the alphabet used by the network. See the comment in data/alphabet.txt for a description of the format.')
    f.DEFINE_string('lm_binary_path', 'data/lm/lm.binary', 'path to the language model binary file created with KenLM')
    f.DEFINE_string('lm_trie_path', 'data/lm/trie', 'path to the language model trie file created with native_client/generate_trie')
    f.DEFINE_integer('beam_width', 1024, 'beam width used in the CTC decoder when building candidate transcriptions')
    f.DEFINE_float('lm_alpha', 0.75, 'the alpha hyperparameter of the CTC decoder. Language Model weight.')
    f.DEFINE_float('lm_beta', 1.85, 'the beta hyperparameter of the CTC decoder. Word insertion weight.')

    # Inference mode

    f.DEFINE_string('one_shot_infer', '', 'one-shot inference mode: specify a wav file and the script will load the checkpoint and perform inference on it.')
