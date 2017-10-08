#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import sys
import scipy.io.wavfile as wav
from deepspeech.model import Model

BEAM_WIDTH = 500
LM_WEIGHT = 2.15
WORD_COUNT_WEIGHT = -0.10
VALID_WORD_COUNT_WEIGHT = 1.10

N_FEATURES = 26
N_CONTEXT = 9

ds = Model(sys.argv[1], N_FEATURES, N_CONTEXT, sys.argv[3])

if len(sys.argv) > 5:
    ds.enableDecoderWithLM(sys.argv[3], sys.argv[4], sys.argv[5], BEAM_WIDTH,
                           LM_WEIGHT, WORD_COUNT_WEIGHT, VALID_WORD_COUNT_WEIGHT)

fs, audio = wav.read(sys.argv[2])
print(ds.stt(audio, fs))
