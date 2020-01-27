#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import argparse
import numpy as np
import wave

from deepspeech import Model


def main():
    parser = argparse.ArgumentParser(description='Running DeepSpeech inference.')
    parser.add_argument('--model', required=True,
                        help='Path to the model (protocol buffer binary file)')
    parser.add_argument('--scorer', nargs='?',
                        help='Path to the external scorer file')
    parser.add_argument('--audio1', required=True,
                        help='First audio file to use in interleaved streams')
    parser.add_argument('--audio2', required=True,
                        help='Second audio file to use in interleaved streams')
    args = parser.parse_args()

    ds = Model(args.model)

    if args.scorer:
        ds.enableExternalScorer(args.scorer)

    fin = wave.open(args.audio1, 'rb')
    fs1 = fin.getframerate()
    audio1 = np.frombuffer(fin.readframes(fin.getnframes()), np.int16)
    fin.close()

    fin = wave.open(args.audio2, 'rb')
    fs2 = fin.getframerate()
    audio2 = np.frombuffer(fin.readframes(fin.getnframes()), np.int16)
    fin.close()

    stream1 = ds.createStream()
    stream2 = ds.createStream()

    splits1 = np.array_split(audio1, 10)
    splits2 = np.array_split(audio2, 10)

    for part1, part2 in zip(splits1, splits2):
        stream1.feedAudioContent(part1)
        stream2.feedAudioContent(part2)

    print(stream1.finishStream())
    print(stream2.finishStream())

if __name__ == '__main__':
    main()
