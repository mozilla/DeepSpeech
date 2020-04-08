#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

import tensorflow.compat.v1 as tfv1
from google.protobuf import text_format


def main():
    # Load and export as string
    with tfv1.gfile.FastGFile(sys.argv[1], "rb") as fin:
        graph_def = tfv1.GraphDef()
        graph_def.ParseFromString(fin.read())

        with tfv1.gfile.FastGFile(sys.argv[1] + "txt", "w") as fout:
            fout.write(text_format.MessageToString(graph_def))


if __name__ == "__main__":
    main()
