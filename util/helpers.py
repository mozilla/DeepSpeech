import tensorflow as tf
import tensorflow.compat.v1 as tfv1
from util.flags import create_flags, FLAGS
from util.logging import log_info, log_error, log_debug, log_progress, create_progressbar
import sys

def load_model(session, checkpoint_filename, drop_source_layers, use_cudnn):
    r'''
    if use_cudnn:
        move all Adam vars to init_vars
    if drop_souce_layers>0:
        move all dropped layers to init_vars
    '''
    
    # Load the checkpoint and put all variables into loading list
    # we will exclude variables we do not wish to load and then
    # we will initialize them instead
    ckpt = tfv1.train.load_checkpoint(checkpoint_filename)
    load_vars = set(tfv1.global_variables())
    init_vars = set()
    if use_cudnn:
        # Identify the variables which we cannot load, and set them
        # for initialization
        for v in load_vars:
            try:
                ckpt.get_tensor(v.op.name)
            except tf.errors.NotFoundError:
                print("CUDNN variable not found:", v.op.name)
                init_vars.add(v)

        load_vars -= init_vars
        
        # Check that the only missing variables (i.e. those to be initialised)
        # are the Adam moment tensors, if they aren't then we have an issue
        if any('Adam' not in v.op.name for v in init_vars):
            log_error('Tried to load a CuDNN RNN checkpoint but there were '
                      'more missing variables than just the Adam moment '
                      'tensors.')
            sys.exit(1)

    if drop_source_layers>0:
        '''
        The transfer learning approach here needs us to supply
        the layers which we will to exclude from the source model.
        Say we want to exclude all layers except for the first one,
        we are dropping five layers total, so:
        drop_source_layers=5
        If we want to use all layers from the source model except the last one, we use this:
        drop_source_layers=1
        '''
        if drop_source_layers>=6:
            log_error('The model only has 6 layers, but you are trying to drop '
                      'all of them or more than all of them. Continuing and '
                      'dropping only 5 layers')
        
        dropped_layers = ['2', '3', 'lstm', '5', '6'][-1 * min(5, int(drop_source_layers)):]

        # Initialize all variables needed for DS, but not loaded from ckpt
        for v in load_vars:
            if any(layer in v.op.name for layer in dropped_layers):
                init_vars.add(v)
                
        load_vars -= init_vars

    for v in init_vars:
        print("Initializing variable from scratch:", v.op.name)
    for v in load_vars:
        print("Loading variable from model:", v.op.name)
        
    init_op = tfv1.variables_initializer(list(init_vars))
    for v in list(load_vars):
        v.load(ckpt.get_tensor(v.op.name), session=session)
    session.run(init_op)


def check_model(checkpoint_filename):
    checkpoint = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir, checkpoint_filename)
    if not checkpoint:
        return False
    return checkpoint.model_checkpoint_path


def try_model(session, load_flag):

    '''
    if auto, cascasde from last --> best --> init
  
    if last, try last, if not found , break
    
    '''
    def try_last():
        return check_model('checkpoint')
        
    def try_best():
        return check_model('best_dev_checkpoint')
        
    def try_init():
        log_info('Initializing variables...')
        # Initialize a new model from scratch
        session.run(tfv1.global_variables_initializer())
        return True
    
    def try_auto():
        res = try_last()
        if not res:
            res = try_best()
            if not res:
                res = try_init()
        return res
    
    checkpoint_path = False
    if load_flag == 'last':
        # returns false or a checkpoint_path
        checkpoint_path = try_last()
        if not checkpoint_path:
            log_error('Unable to load LAST model from specified checkpoint dir'
                      ' - consider using load option "auto" or "init".')
            sys.exit(1)
            
    if load_flag == 'best':
        # returns false or a checkpoint path
        checkpoint_path = try_best()
        if not checkpoint_path:
            log_error('Unable to load BEST model from specified checkpoint dir'
                      ' - consider using load option "auto" or "init".')
            sys.exit(1)
            
    if load_flag == 'init':
        # no return
        try_init()
        
    if load_flag == 'auto':
        # returns true or a checkpoint path
        checkpoint_path = try_auto()

    if not checkpoint_path in [True,False]:
        load_model(session, checkpoint_path, FLAGS.drop_source_layers, FLAGS.use_cudnn_rnn)




def keep_only_digits(txt):
    return ''.join(filter(lambda c: c.isdigit(), txt))

def secs_to_hours(secs):
    hours, remainder = divmod(secs, 3600)
    minutes, seconds = divmod(remainder, 60)
    return '%d:%02d:%02d' % (hours, minutes, seconds)
