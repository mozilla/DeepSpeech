#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

import tensorflow.compat.v1 as tfv1


def main():
    with tfv1.gfile.FastGFile(sys.argv[1], "rb") as fin:
        graph_def = tfv1.GraphDef()
        graph_def.ParseFromString(fin.read())

        print("\n".join(sorted(set(n.op for n in graph_def.node))))


if __name__ == "__main__":
    main()
