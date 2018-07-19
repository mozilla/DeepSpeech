#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import sys

# Load and export as string
with tf.gfile.FastGFile(sys.argv[1], 'rb') as fin:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(fin.read())

    with tf.gfile.FastGFile(sys.argv[1] + 'txt', 'w') as fout:
        from google.protobuf import text_format
        fout.write(text_format.MessageToString(graph_def))
