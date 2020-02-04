import tensorflow as tf
import tensorflow.compat.v1 as tfv1
from util.flags import create_flags, FLAGS
from util.logging import log_info, log_error, log_debug, log_progress, create_progressbar
import sys

def load_model(session, checkpoint_filename, drop_source_layers, load_cudnn, train=True):
    # Load the checkpoint and put all variables into loading list
    # we will exclude variables we do not wish to load and then
    # we will initialize them instead
    ckpt = tfv1.train.load_checkpoint(checkpoint_filename)
    load_vars = set(tfv1.global_variables())
    init_vars = set()
    
    if load_cudnn:
        # Initialize training from a CuDNN RNN checkpoint
        # Identify the variables which we cannot load, and set them
        # for initialization
        for v in load_vars:
            try:
                ckpt.get_tensor(v.op.name)
            except tf.errors.NotFoundError:
                log_error('CUDNN variable not found: %s' % (v.op.name))
                init_vars.add(v)

        load_vars -= init_vars
        
        # Check that the only missing variables (i.e. those to be initialised)
        # are the Adam moment tensors, if they aren't then we have an issue
        if any('Adam' not in v.op.name for v in init_vars):
            log_error('Tried to load a CuDNN RNN checkpoint but there were '
                      'more missing variables than just the Adam moment '
                      'tensors.')
            sys.exit(1)

    if train and drop_source_layers > 0:
        # This transfer learning approach requires supplying
        # the layers which we exclude from the source model.
        # Say we want to exclude all layers except for the first one,
        # then we are dropping five layers total, so: drop_source_layers=5
        # If we want to use all layers from the source model except
        # the last one, we use this: drop_source_layers=1
        if drop_source_layers >= 6:
            log_error('The checkpoint only has 6 layers, but you are trying to drop '
                      'all of them or more than all of them. Continuing and '
                      'dropping only 5 layers')
            drop_source_layers = 5
        
        dropped_layers = ['2', '3', 'lstm', '5', '6'][-1 * int(drop_source_layers):]
        # Initialize all variables needed for DS, but not loaded from ckpt
        for v in load_vars:
            if any(layer in v.op.name for layer in dropped_layers):
                init_vars.add(v)
        load_vars -= init_vars

    for v in init_vars:
        log_info('Initializing variable from scratch: %s' % (v.op.name))
    for v in load_vars:
        log_info('Loading variable from checkpoint: %s' % (v.op.name))
        
    init_op = tfv1.variables_initializer(list(init_vars))
    for v in list(load_vars):
        v.load(ckpt.get_tensor(v.op.name), session=session)
    session.run(init_op)


def check_model(checkpoint_dir, checkpoint_filename):
    r'''
    returns None or a checkpoint path
    '''
    checkpoint = tf.train.get_checkpoint_state(checkpoint_dir, checkpoint_filename)
    if not checkpoint:
        return None
    return checkpoint.model_checkpoint_path


def try_model(session, checkpoint_dir, load_flag, train=True):
    
    def try_last():
        log_info('Trying to load last saved checkpoint')
        return check_model(checkpoint_dir, 'checkpoint')
    
    def try_best():
        log_info('Trying to load best saved checkpoint')
        return check_model(checkpoint_dir, 'best_dev_checkpoint')
    
    def try_init():
        log_info('Initializing variables from scratch.')
        session.run(tfv1.global_variables_initializer())
        return None
    
    def try_auto(train):
        res = try_last()
        if not res:
            res = try_best()
            if train and not res:
                res = try_init()
            else:
                log_error('Tried to load a checkpoint (for something other than training) '
                          'but could not find it. Exiting now. ')
                sys.exit(1)
        return res
    
    checkpoint_path = None
    if load_flag == 'last':
        checkpoint_path = try_last()
        if not checkpoint_path:
            log_error('Unable to load LAST model from specified checkpoint directory %s.' % (checkpoint_dir))
            sys.exit(1)
    
    if load_flag == 'best':
        checkpoint_path = try_best()
        if not checkpoint_path:
            log_error('Unable to load BEST model from specified checkpoint directory %s.' % (checkpoint_dir))
            sys.exit(1)
    
    if load_flag == 'init':
        try_init()
    
    if load_flag == 'auto':
        checkpoint_path = try_auto(train)
    
    if checkpoint_path:
        load_model(session, checkpoint_path, FLAGS.drop_source_layers, FLAGS.load_cudnn, train)


def keep_only_digits(txt):
    return ''.join(filter(lambda c: c.isdigit(), txt))


def secs_to_hours(secs):
    hours, remainder = divmod(secs, 3600)
    minutes, seconds = divmod(remainder, 60)
    return '%d:%02d:%02d' % (hours, minutes, seconds)

# pylint: disable=import-outside-toplevel
def check_ctcdecoder_version():
    import sys
    import os
    import semver

    ds_version_s = open(os.path.join(os.path.dirname(__file__), '../VERSION')).read().strip()

    try:
        from ds_ctcdecoder import __version__ as decoder_version
    except ImportError as e:
        if e.msg.find('__version__') > 0:
            print("DeepSpeech version ({ds_version}) requires CTC decoder to expose __version__. Please upgrade the ds_ctcdecoder package to version {ds_version}".format(ds_version=ds_version_s))
            sys.exit(1)
        raise e

    decoder_version_s = decoder_version.decode()

    rv = semver.compare(ds_version_s, decoder_version_s)
    if rv != 0:
        print("DeepSpeech version ({}) and CTC decoder version ({}) do not match. Please ensure matching versions are in use.".format(ds_version_s, decoder_version_s))
        sys.exit(1)

    return rv
