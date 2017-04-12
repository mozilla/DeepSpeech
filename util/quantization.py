#!/usr/bin/env python
from __future__ import print_function
from __future__ import absolute_import
# -*- coding: utf-8 -*-

## USAGE:
### Train and quantize:
### $ ./bin/run-ldc93s1.sh --quantize_model "test.test.ldc93s1.pb" --apply_transforms "quantize_weights"
##
### Test quantization:
### $ ./bin/run_quantization.py --run_quantized_model "test.test.ldc93s1.pb"

import os
import subprocess
import tempfile

import tensorflow as tf
FLAGS = tf.app.flags.FLAGS

from util.tc import maybe_download_tc_bin

def quantization_flags():
    tf.app.flags.DEFINE_string  ('quantize_model', '', 'Enable quantization within this ProtocolBuffer binary file')
    tf.app.flags.DEFINE_string  ('apply_transforms', '', 'Transforms to apply to the graph')

    # Transforms are:
    #
    # add_default_attributes
    # fold_batch_norms
    # fold_constants
    # fold_old_batch_norms
    # freeze_requantization_ranges
    # fuse_pad_and_conv
    # fuse_resize_and_conv
    # fuse_resize_pad_and_conv
    # insert_logging
    # merge_duplicate_nodes
    # obsfucate_names
    # quantize_nodes
    # quantize_weights
    # remove_attribute
    # remove_device
    # remove_nodes
    # rename_attribute
    # rename_op
    # round_weights
    # set_device
    # sort_by_execution_order
    # strip_unused_nodes

def exec_freeze_model(inGraph, outGraph, nodes, checkpoint_dir, clearDevices=True):
    from tensorflow.python.tools import freeze_graph as freezer

    rv = -1
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        # Let us re-use freeze_graph.py directly, but we need to dump graphdef as a file anyway
        frozen = freezer.freeze_graph(
            input_graph=inGraph,
            input_saver="",
            input_checkpoint=ckpt.model_checkpoint_path,
            output_graph=outGraph,
            input_binary=True,
            output_node_names=nodes,
            restore_op_name="save/restore_all",
            filename_tensor_name="save/Const:0",
            clear_devices=clearDevices,
            initializer_nodes="")
        rv = 0

    return rv

def exec_transform_quantize_model(frozen, out, nodes_input, nodes_output):

    maybe_download_tc_bin(target_dir='./bin/',
                          tc_url='https://index.taskcluster.net/v1/task/project.deepspeech.tensorflow.pip.v1.0.0-warpctc.cpu/artifacts/public/transform_graph',
                          progress=True)

    args = [
        'bin/transform_graph',
        '--in_graph=%s'   % frozen,
        '--out_graph=%s'  % out,
        '--inputs=%s'     % nodes_input,
        '--outputs=%s'    % nodes_output,
        '--transforms=%s' % FLAGS.apply_transforms
    ]

    # build with:
    #  bazel build -c opt --copt=-mtune=generic --cxxopt=-mtune=generic --copt=-march=x86-64 --cxxopt=-march=x86-64 --copt=-msse --cxxopt=-msse --copt=-msse2 --cxxopt=-msse2 --copt=-msse3 --cxxopt=-msse3 --copt=-msse4.1 --cxxopt=-msse4.1 --copt=-msse4.2 --cxxopt=-msse4.2 --copt=-mavx --cxxopt=-mavx --copt=-mavx2 --cxxopt=-mavx2 --copt=-mfma --cxxopt=-mfma --verbose_failures tensorflow/tools/graph_transforms:transform_graph
    #  cp [...]/bazel-bin/tensorflow/tools/graph_transforms/transform_graph bin/ && chmod +x bin/transform_graph && upx -6 bin/transform_graph
    #  tested with upstream UPX v3.93 (3.91 from Ubuntu 16.10 not working)
    rv = subprocess.check_call(args)

    return rv

def exec_with_cleanup(*args): #, op=None, inGraph=None, outGraph=None):
    args = list(args)
    args.reverse()

    op  = args.pop()
    inG = args.pop()
    ouG = args.pop()

    args.reverse()
    args = tuple(args)

    _globals = globals()
    if not op in _globals:
        print("No", op, "exists, cannot call")
        assert False

    fun = globals()[op]

    rv = fun(inG, ouG, *args)
    if rv == 0:
        print("Successfull call of:", op, "produced", ouG)
    else:
        print("Error while calling:", op)

    os.unlink(inG)

    return rv == 0

def get_input_output_names():
    inputNames   =  "input_node"
    outputNames  =  "output_node"

    return inputNames, outputNames

def open_temp_pbfile(basename=None):
    fd, name  = tempfile.mkstemp(prefix=basename, suffix='.pb', text=False)
    os.close(fd)
    return name

def quantize_model(graph_source, checkpoint_dir):
    tmpGraphDef     = open_temp_pbfile('tmpGraphDef')
    frozenOut       = open_temp_pbfile('tmpGraphFrozen')
    finalOut        = os.path.abspath(FLAGS.quantize_model)

    tmpGraphDefDir  = os.path.dirname(tmpGraphDef)
    tmpGraphDefFile = os.path.basename(tmpGraphDef)

    inputNames, outputNames = get_input_output_names()

    # Freeze & export GraphDef
    print("Freezing model using current checkpoint")
    tf.train.write_graph(graph_source.graph.as_graph_def(), tmpGraphDefDir, tmpGraphDefFile, as_text=False)

    if len(FLAGS.apply_transforms) > 0:
        exec_with_cleanup('exec_freeze_model', tmpGraphDef, frozenOut, outputNames, checkpoint_dir)
        exec_with_cleanup('exec_transform_quantize_model', frozenOut, finalOut, inputNames, outputNames)
    else:
        exec_with_cleanup('exec_freeze_model', tmpGraphDef, finalOut, outputNames, checkpoint_dir)

    return

def print_time(d):
    print('------------------------' +
          '\n' + 'Inference: ' + format_duration(d) + '\n' +
          '------------------------')

def format_duration(duration):
    '''Formats the result of an even stopwatch call as seconds.microseconds'''
    return '%s' % duration.total_seconds()

def do_inference(batch_set, sess, logits):
    # Obtain the next batch of data
    batch_x, batch_seq_len, batch_y = batch_set.next_batch()

    #print("batch_x=", batch_x)
    #print("batch_seq_len=", batch_seq_len)
    #print("batch_y=", batch_y)

    batch_x = sess.run(batch_x, feed_dict={ batch_set._queue_selector: 0 })
    batch_seq_len = sess.run(batch_seq_len, feed_dict={ batch_set._queue_selector: 0 })
    # batch_y = sess.run(batch_y)

    #print("got batch_x=", batch_x)
    #print("got batch_seq_len=", batch_seq_len)
    #print("got batch_y=", batch_y)

    g = tf.get_default_graph()

    ###run inference
    #print("got logits=", type(logits), logits)

    input_tensor   = g.get_tensor_by_name("input_node:0")
    output_tensor  = g.get_tensor_by_name("logits_output_node:0")
    input_lengths  = g.get_tensor_by_name("input_lengths:0")

    #print("input_tensor=", input_tensor)
    #print("output_tensor=", output_tensor)
    #print("input_lengths=", input_lengths)

    #print("running session")
    logits = sess.run( output_tensor, feed_dict={ input_tensor: batch_x, input_lengths: batch_seq_len })
    #print("SESSION RUN: output_tensor=", output_tensor)
    #print("SESSION RUN: logits=", logits, logits.shape)

    #print("SEQ LENGTH", batch_seq_len, "shape:", batch_seq_len.shape)
    #print("LABELS=", batch_y)

    total_loss = tf.nn.ctc_loss(labels=batch_y, inputs=logits, sequence_length=batch_seq_len)

    # Calculate the average loss across the batch
    avg_loss = tf.reduce_mean(total_loss)

    # Beam search decode the batch
    decoded, _ = tf.nn.ctc_beam_search_decoder(logits, batch_seq_len, merge_repeated=False)

    # Compute the edit (Levenshtein) distance
    distance = tf.edit_distance(tf.cast(decoded[0], tf.int32), batch_y)

    # Compute the accuracy
    accuracy = tf.reduce_mean(distance)

    # Finally we return the
    # - calculated total and
    # - average losses,
    # - the Levenshtein distance,
    # - the recognition accuracy,
    # - the decoded batch and
    # - the original batch_y (which contains the verified transcriptions).
    return total_loss, avg_loss, distance, accuracy, decoded, batch_y
