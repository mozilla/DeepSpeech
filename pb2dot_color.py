#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import re
import os
import sys

from tensorflow.core.framework import graph_pb2

"""
Takes binary protocolbuffer graph on stdin and
produce colored dot representation on stdout
"""

graph = graph_pb2.GraphDef()
graph.ParseFromString(sys.stdin.read())

buckets = {}

def try_add_to_bucket(bucket_name, node_name):
    try:
        buckets[bucket_name].append(node_name)
    except KeyError as e:
        buckets[bucket_name] = [ node_name ]

print("digraph graphname {", file=sys.stdout)
for node in graph.node:
    output_name = node.name
    color = ""

    if re.match(r"^garbage_", output_name):
        color = "style=filled color=cyan fillcolor=cyan"
        try_add_to_bucket('main', output_name)

    if re.match(r"^bidirectional_rnn\/fw\/fw", output_name):
        color = "style=filled color=chartreuse1 fillcolor=chartreuse1"
        try_add_to_bucket('rnn_fw', output_name)
    elif re.match(r"^bidirectional_rnn\/fw\/basic_rnn_cell", output_name):
        color = "style=filled color=chartreuse2 fillcolor=chartreuse3"
        try_add_to_bucket('rnn_fw', output_name)
    elif re.match(r"^bidirectional_rnn\/fw", output_name):
        color = "style=filled color=chartreuse fillcolor=chartreuse"
        try_add_to_bucket('rnn_fw', output_name)

    if re.match(r"^bidirectional_rnn\/bw\/bw", output_name):
        color = "style=filled color=chocolate1 fillcolor=chocolate1"
        try_add_to_bucket('rnn_bw', output_name)
    elif re.match(r"^bidirectional_rnn\/bw\/basic_rnn_cell", output_name):
        color = "style=filled color=chocolate2 fillcolor=chocolate3"
        try_add_to_bucket('rnn_bw', output_name)
    elif re.match(r"^bidirectional_rnn\/bw", output_name):
        color = "style=filled color=chocolate fillcolor=chocolate"
        try_add_to_bucket('rnn_bw', output_name)

    print("  \"" + output_name + "\" [label=\"" + node.op + "\n" + output_name + "\", " + color + "];", file=sys.stdout)

    for input_full_name in node.input:
        parts = input_full_name.split(":")
        input_name = re.sub(r"^\^", "", parts[0])
        print("  \"" + input_name + "\" -> \"" + output_name + "\";", file=sys.stdout)
        try_add_to_bucket('edges', input_name + "->" + output_name)

all_buckets = 0
for b in buckets.keys():
    all_buckets += len(buckets[b])
    print("// bucket_name=%s, len=%d" % (b, len(buckets[b])), file=sys.stdout)
print("// all_buckets=%d" % (all_buckets), file=sys.stdout)

print("}", file=sys.stdout)
