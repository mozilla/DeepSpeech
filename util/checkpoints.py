import sys
import tensorflow as tf
import tensorflow.compat.v1 as tfv1

from util.flags import FLAGS
from util.logging import log_info, log_error, log_warn


def _load_checkpoint(session, checkpoint_path):
    # Load the checkpoint and put all variables into loading list
    # we will exclude variables we do not wish to load and then
    # we will initialize them instead
    ckpt = tfv1.train.load_checkpoint(checkpoint_path)
    load_vars = set(tfv1.global_variables())
    init_vars = set()

    if FLAGS.load_cudnn:
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
        init_var_names = [v.op.name for v in init_vars]
        if any('Adam' not in v for v in init_var_names):
            log_error('Tried to load a CuDNN RNN checkpoint but there were '
                      'more missing variables than just the Adam moment '
                      'tensors. Missing variables: {}'.format(init_var_names))
            sys.exit(1)

    if FLAGS.drop_source_layers > 0:
        # This transfer learning approach requires supplying
        # the layers which we exclude from the source model.
        # Say we want to exclude all layers except for the first one,
        # then we are dropping five layers total, so: drop_source_layers=5
        # If we want to use all layers from the source model except
        # the last one, we use this: drop_source_layers=1
        if FLAGS.drop_source_layers >= 6:
            log_warn('The checkpoint only has 6 layers, but you are trying to drop '
                     'all of them or more than all of them. Continuing and '
                     'dropping only 5 layers.')
            FLAGS.drop_source_layers = 5

        dropped_layers = ['2', '3', 'lstm', '5', '6'][-1 * int(FLAGS.drop_source_layers):]
        # Initialize all variables needed for DS, but not loaded from ckpt
        for v in load_vars:
            if any(layer in v.op.name for layer in dropped_layers):
                init_vars.add(v)
        load_vars -= init_vars

    for v in load_vars:
        log_info('Loading variable from checkpoint: %s' % (v.op.name))
        v.load(ckpt.get_tensor(v.op.name), session=session)

    for v in init_vars:
        log_info('Initializing variable: %s' % (v.op.name))
        session.run(v.initializer)


def _checkpoint_path_or_none(checkpoint_filename):
    checkpoint = tfv1.train.get_checkpoint_state(FLAGS.load_checkpoint_dir, checkpoint_filename)
    if not checkpoint:
        return None
    return checkpoint.model_checkpoint_path


def _initialize_all_variables(session):
    init_vars = tfv1.global_variables()
    for v in init_vars:
        session.run(v.initializer)


def load_or_init_graph(session, method_order):
    '''
    Load variables from checkpoint or initialize variables following the method
    order specified in the method_order parameter.

    Valid methods are 'best', 'last' and 'init'.
    '''
    for method in method_order:
        # Load best validating checkpoint, saved in checkpoint file 'best_dev_checkpoint'
        if method == 'best':
            ckpt_path = _checkpoint_path_or_none('best_dev_checkpoint')
            if ckpt_path:
                log_info('Loading best validating checkpoint from {}'.format(ckpt_path))
                return _load_checkpoint(session, ckpt_path)
            log_info('Could not find best validating checkpoint.')

        # Load most recent checkpoint, saved in checkpoint file 'checkpoint'
        elif method == 'last':
            ckpt_path = _checkpoint_path_or_none('checkpoint')
            if ckpt_path:
                log_info('Loading most recent checkpoint from {}'.format(ckpt_path))
                return _load_checkpoint(session, ckpt_path)
            log_info('Could not find most recent checkpoint.')

        # Initialize all variables
        elif method == 'init':
            log_info('Initializing all variables.')
            return _initialize_all_variables(session)

        else:
            log_error('Unknown initialization method: {}'.format(method))
            sys.exit(1)

    log_error('All initialization methods failed ({}).'.format(method_order))
    sys.exit(1)
