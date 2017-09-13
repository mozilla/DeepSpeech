#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import tensorflow as tf
import re
import os
import sys

if len(sys.argv) != 2:
    print("%s: graphName" % (sys.argv[0]))
    print("will produce: graphName.pb and graphName.dot")
    sys.exit(1)

graph_pb  = os.path.abspath("%s.pb" % sys.argv[1])
graph_dot = os.path.abspath("%s.dot" % sys.argv[1])

g = tf.Graph()
with g.as_default() as g:
    n_input    = 26
    n_context  = 9
    n_cell_dim = 2
    n_hidden_3 = 2 * n_cell_dim

    batch_x = tf.placeholder(tf.int32, [None, None, n_input + 2*n_input*n_context], name='garbage_batch_x')
    seq_length = tf.placeholder(tf.int32, [None], name='garbage_seq_length')

    batch_x_shape = tf.shape(batch_x, name='garbage_batch_x_shape')
    batch_x = tf.transpose(batch_x, [1, 0, 2], name='garbage_batch_x_transpose')
    batch_x = tf.reshape(batch_x, [-1, n_input + 2*n_input*n_context], name='garbage_batch_x_reshape')
    layer_input = tf.reshape(batch_x, [-1, batch_x_shape[0], n_hidden_3], name='garbage_layer_input')

    ##layer_input = tf.fake_quant_with_min_max_args(layer_input, min=0.0, max=255.0)

    rnn1 = tf.contrib.rnn.BasicRNNCell(n_cell_dim)
    rnn2 = tf.contrib.rnn.BasicRNNCell(n_cell_dim)

    outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=rnn1,
                                                             cell_bw=rnn2,
                                                             inputs=layer_input,
                                                             dtype=tf.int32,
                                                             time_major=True,
                                                             sequence_length=seq_length)

    outputs = tf.concat(outputs, 2, name='garbage_outputs')

with tf.Session(graph=g) as sess:
    tf.global_variables_initializer().run()
    r = sess.run({})
    tf.train.write_graph(g, os.path.dirname(graph_pb), os.path.basename(graph_pb), as_text=False)
