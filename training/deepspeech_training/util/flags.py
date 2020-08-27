from __future__ import absolute_import, division, print_function

import os
import absl.flags

FLAGS = absl.flags.FLAGS

# sphinx-doc: training_ref_flags_start
def create_flags():
    # Importer
    # ========

    f = absl.flags

    f.DEFINE_string('train_files', '', 'comma separated list of files specifying the dataset used for training. Multiple files will get merged. If empty, training will not be run.')
    f.DEFINE_string('dev_files', '', 'comma separated list of files specifying the datasets used for validation. Multiple files will get reported separately. If empty, validation will not be run.')
    f.DEFINE_string('test_files', '', 'comma separated list of files specifying the datasets used for testing. Multiple files will get reported separately. If empty, the model will not be tested.')
    f.DEFINE_string('metrics_files', '', 'comma separated list of files specifying the datasets used for tracking of metrics (after validation step). Currently the only metric is the CTC loss but without affecting the tracking of best validation loss. Multiple files will get reported separately. If empty, metrics will not be computed.')

    f.DEFINE_string('read_buffer', '1MB', 'buffer-size for reading samples from datasets (supports file-size suffixes KB, MB, GB, TB)')
    f.DEFINE_string('feature_cache', '', 'cache MFCC features to disk to speed up future training runs on the same data. This flag specifies the path where cached features extracted from --train_files will be saved. If empty, or if online augmentation flags are enabled, caching will be disabled.')
    f.DEFINE_integer('cache_for_epochs', 0, 'after how many epochs the feature cache is invalidated again - 0 for "never"')

    f.DEFINE_integer('feature_win_len', 32, 'feature extraction audio window length in milliseconds')
    f.DEFINE_integer('feature_win_step', 20, 'feature extraction window step length in milliseconds')
    f.DEFINE_integer('audio_sample_rate', 16000, 'sample rate value expected by model')

    # Data Augmentation
    # ================

    f.DEFINE_multi_string('augment', None, 'specifies an augmentation of the training samples. Format is "--augment operation[param1=value1, ...]"')

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

    # Performance

    f.DEFINE_integer('inter_op_parallelism_threads', 0, 'number of inter-op parallelism threads - see tf.ConfigProto for more details. USE OF THIS FLAG IS UNSUPPORTED')
    f.DEFINE_integer('intra_op_parallelism_threads', 0, 'number of intra-op parallelism threads - see tf.ConfigProto for more details. USE OF THIS FLAG IS UNSUPPORTED')
    f.DEFINE_boolean('use_allow_growth', False, 'use Allow Growth flag which will allocate only required amount of GPU memory and prevent full allocation of available GPU memory')
    f.DEFINE_boolean('load_cudnn', False, 'Specifying this flag allows one to convert a CuDNN RNN checkpoint to a checkpoint capable of running on a CPU graph.')
    f.DEFINE_boolean('train_cudnn', False, 'use CuDNN RNN backend for training on GPU. Note that checkpoints created with this flag can only be used with CuDNN RNN, i.e. fine tuning on a CPU device will not work')
    f.DEFINE_boolean('automatic_mixed_precision', False, 'whether to allow automatic mixed precision training. USE OF THIS FLAG IS UNSUPPORTED. Checkpoints created with automatic mixed precision training will not be usable without mixed precision.')

    # Sample limits

    f.DEFINE_integer('limit_train', 0, 'maximum number of elements to use from train set - 0 means no limit')
    f.DEFINE_integer('limit_dev', 0, 'maximum number of elements to use from validation set - 0 means no limit')
    f.DEFINE_integer('limit_test', 0, 'maximum number of elements to use from test set - 0 means no limit')

    # Sample order

    f.DEFINE_boolean('reverse_train', False, 'if to reverse sample order of the train set')
    f.DEFINE_boolean('reverse_dev', False, 'if to reverse sample order of the dev set')
    f.DEFINE_boolean('reverse_test', False, 'if to reverse sample order of the test set')

    # Checkpointing

    f.DEFINE_string('checkpoint_dir', '', 'directory from which checkpoints are loaded and to which they are saved - defaults to directory "deepspeech/checkpoints" within user\'s data home specified by the XDG Base Directory Specification')
    f.DEFINE_string('load_checkpoint_dir', '', 'directory in which checkpoints are stored - defaults to directory "deepspeech/checkpoints" within user\'s data home specified by the XDG Base Directory Specification')
    f.DEFINE_string('save_checkpoint_dir', '', 'directory to which checkpoints are saved - defaults to directory "deepspeech/checkpoints" within user\'s data home specified by the XDG Base Directory Specification')
    f.DEFINE_integer('checkpoint_secs', 600, 'checkpoint saving interval in seconds')
    f.DEFINE_integer('max_to_keep', 5, 'number of checkpoint files to keep - default value is 5')
    f.DEFINE_string('load_train', 'auto', 'what checkpoint to load before starting the training process. "last" for loading most recent epoch checkpoint, "best" for loading best validation loss checkpoint, "init" for initializing a new checkpoint, "auto" for trying several options.')
    f.DEFINE_string('load_evaluate', 'auto', 'what checkpoint to load for evaluation tasks (test epochs, model export, single file inference, etc). "last" for loading most recent epoch checkpoint, "best" for loading best validation loss checkpoint, "auto" for trying several options.')

    # Transfer Learning

    f.DEFINE_integer('drop_source_layers', 0, 'single integer for how many layers to drop from source model (to drop just output == 1, drop penultimate and output ==2, etc)')

    # Exporting

    f.DEFINE_string('export_dir', '', 'directory in which exported models are stored - if omitted, the model won\'t get exported')
    f.DEFINE_boolean('remove_export', False, 'whether to remove old exported models')
    f.DEFINE_boolean('export_tflite', False, 'export a graph ready for TF Lite engine')
    f.DEFINE_integer('n_steps', 16, 'how many timesteps to process at once by the export graph, higher values mean more latency')
    f.DEFINE_boolean('export_zip', False, 'export a TFLite model and package with LM and info.json')
    f.DEFINE_string('export_file_name', 'output_graph', 'name for the exported model file name')
    f.DEFINE_integer('export_beam_width', 500, 'default beam width to embed into exported graph')

    # Model metadata

    f.DEFINE_string('export_author_id', 'author', 'author of the exported model. GitHub user or organization name used to uniquely identify the author of this model')
    f.DEFINE_string('export_model_name', 'model', 'name of the exported model. Must not contain forward slashes.')
    f.DEFINE_string('export_model_version', '0.0.1', 'semantic version of the exported model. See https://semver.org/. This is fully controlled by you as author of the model and has no required connection with DeepSpeech versions')

    def str_val_equals_help(name, val_desc):
        f.DEFINE_string(name, '<{}>'.format(val_desc), val_desc)

    str_val_equals_help('export_contact_info', 'public contact information of the author. Can be an email address, or a link to a contact form, issue tracker, or discussion forum. Must provide a way to reach the model authors')
    str_val_equals_help('export_license', 'SPDX identifier of the license of the exported model. See https://spdx.org/licenses/. If the license does not have an SPDX identifier, use the license name.')
    str_val_equals_help('export_language', 'language the model was trained on - IETF BCP 47 language tag including at least language, script and region subtags. E.g. "en-Latn-UK" or "de-Latn-DE" or "cmn-Hans-CN". Include as much info as you can without loss of precision. For example, if a model is trained on Scottish English, include the variant subtag: "en-Latn-GB-Scotland".')
    str_val_equals_help('export_min_ds_version', 'minimum DeepSpeech version (inclusive) the exported model is compatible with')
    str_val_equals_help('export_max_ds_version', 'maximum DeepSpeech version (inclusive) the exported model is compatible with')
    str_val_equals_help('export_description', 'Freeform description of the model being exported. Markdown accepted. You can also leave this flag unchanged and edit the generated .md file directly. Useful things to describe are demographic and acoustic characteristics of the data used to train the model, any architectural changes, names of public datasets that were used when applicable, hyperparameters used for training, evaluation results on standard benchmark datasets, etc.')

    # Reporting

    f.DEFINE_integer('log_level', 1, 'log level for console logs - 0: DEBUG, 1: INFO, 2: WARN, 3: ERROR')
    f.DEFINE_boolean('show_progressbar', True, 'Show progress for training, validation and testing processes. Log level should be > 0.')

    f.DEFINE_boolean('log_placement', False, 'whether to log device placement of the operators to the console')
    f.DEFINE_integer('report_count', 5, 'number of phrases for each of best WER, median WER and worst WER to print out during a WER report')

    f.DEFINE_string('summary_dir', '', 'target directory for TensorBoard summaries - defaults to directory "deepspeech/summaries" within user\'s data home specified by the XDG Base Directory Specification')

    f.DEFINE_string('test_output_file', '', 'path to a file to save all src/decoded/distance/loss tuples generated during a test epoch')

    # Geometry

    f.DEFINE_integer('n_hidden', 2048, 'layer width to use when initialising layers')
    f.DEFINE_boolean('layer_norm', False, 'wether to use layer-normalization after each fully-connected layer (except the last one)')

    # Initialization

    f.DEFINE_integer('random_seed', 4568, 'default random seed that is used to initialize variables')

    # Early Stopping

    f.DEFINE_boolean('early_stop', False, 'Enable early stopping mechanism over validation dataset. If validation is not being run, early stopping is disabled.')
    f.DEFINE_integer('es_epochs', 25, 'Number of epochs with no improvement after which training will be stopped. Loss is not stored in the checkpoint so when checkpoint is revived it starts the loss calculation from start at that point')
    f.DEFINE_float('es_min_delta', 0.05, 'Minimum change in loss to qualify as an improvement. This value will also be used in Reduce learning rate on plateau')

    # Reduce learning rate on plateau

    f.DEFINE_boolean('reduce_lr_on_plateau', False, 'Enable reducing the learning rate if a plateau is reached. This is the case if the validation loss did not improve for some epochs.')
    f.DEFINE_integer('plateau_epochs', 10, 'Number of epochs to consider for RLROP. Has to be smaller than es_epochs from early stopping')
    f.DEFINE_float('plateau_reduction', 0.1, 'Multiplicative factor to apply to the current learning rate if a plateau has occurred.')
    f.DEFINE_boolean('force_initialize_learning_rate', False, 'Force re-initialization of learning rate which was previously reduced.')

    # Decoder

    f.DEFINE_boolean('utf8', False, 'enable UTF-8 mode. When this is used the model outputs UTF-8 sequences directly rather than using an alphabet mapping.')
    f.DEFINE_string('alphabet_config_path', 'data/alphabet.txt', 'path to the configuration file specifying the alphabet used by the network. See the comment in data/alphabet.txt for a description of the format.')
    f.DEFINE_string('scorer_path', '', 'path to the external scorer file.')
    f.DEFINE_alias('scorer', 'scorer_path')
    f.DEFINE_integer('beam_width', 1024, 'beam width used in the CTC decoder when building candidate transcriptions')
    f.DEFINE_float('lm_alpha', 0.931289039105002, 'the alpha hyperparameter of the CTC decoder. Language Model weight.')
    f.DEFINE_float('lm_beta', 1.1834137581510284, 'the beta hyperparameter of the CTC decoder. Word insertion weight.')
    f.DEFINE_float('cutoff_prob', 1.0, 'only consider characters until this probability mass is reached. 1.0 = disabled.')
    f.DEFINE_integer('cutoff_top_n', 300, 'only process this number of characters sorted by probability mass for each time step. If bigger than alphabet size, disabled.')

    # Inference mode

    f.DEFINE_string('one_shot_infer', '', 'one-shot inference mode: specify a wav file and the script will load the checkpoint and perform inference on it.')

    # Optimizer mode

    f.DEFINE_float('lm_alpha_max', 5, 'the maximum of the alpha hyperparameter of the CTC decoder explored during hyperparameter optimization. Language Model weight.')
    f.DEFINE_float('lm_beta_max', 5, 'the maximum beta hyperparameter of the CTC decoder explored during hyperparameter optimization. Word insertion weight.')
    f.DEFINE_integer('n_trials', 2400, 'the number of trials to run during hyperparameter optimization.')

    # Register validators for paths which require a file to be specified

    f.register_validator('alphabet_config_path',
                         os.path.isfile,
                         message='The file pointed to by --alphabet_config_path must exist and be readable.')

    f.register_validator('one_shot_infer',
                         lambda value: not value or os.path.isfile(value),
                         message='The file pointed to by --one_shot_infer must exist and be readable.')

# sphinx-doc: training_ref_flags_end
