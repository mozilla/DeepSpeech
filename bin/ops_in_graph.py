#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import sys

with tf.gfile.FastGFile(sys.argv[1], 'rb') as fin:
    graph_def = tf.GraphDef()
    graph_def.MergeFromString(fin.read())

    print('\n'.join(sorted(set(n.op for n in graph_def.node))))
