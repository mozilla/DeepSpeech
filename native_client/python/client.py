#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import argparse
import numpy as np
import shlex
import subprocess
import sys
import wave

from deepspeech.model import Model, print_versions
from timeit import default_timer as timer

try:
    from shhlex import quote
except ImportError:
    from pipes import quote

# These constants control the beam search decoder

# Beam width used in the CTC decoder when building candidate transcriptions
BEAM_WIDTH = 500

# The alpha hyperparameter of the CTC decoder. Language Model weight
LM_WEIGHT = 1.75

# The beta hyperparameter of the CTC decoder. Word insertion weight (penalty)
WORD_COUNT_WEIGHT = 1.00

# Valid word insertion weight. This is used to lessen the word insertion penalty
# when the inserted word is part of the vocabulary
VALID_WORD_COUNT_WEIGHT = 1.00


# These constants are tied to the shape of the graph used (changing them changes
# the geometry of the first layer), so make sure you use the same constants that
# were used during training

# Number of MFCC features to use
N_FEATURES = 26

# Size of the context window used for producing timesteps in the input vector
N_CONTEXT = 9

def convert_samplerate(audio_path):
    sox_cmd = 'sox {} --type raw --bits 16 --channels 1 --rate 16000 - '.format(quote(audio_path))
    try:
        output = subprocess.check_output(shlex.split(sox_cmd), stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise RuntimeError('SoX returned non-zero status: {}'.format(e.stderr))
    except OSError as e:
        raise OSError(e.errno, 'SoX not found, use 16kHz files or install it: {}'.format(e.strerror))

    return 16000, np.frombuffer(output, np.int16)

def main():
    parser = argparse.ArgumentParser(description='Running DeepSpeech inference.')
    parser.add_argument('--model',
                        help='Path to the model (protocol buffer binary file)')
    parser.add_argument('--alphabet',
                        help='Path to the configuration file specifying the alphabet used by the network')
    parser.add_argument('--lm', nargs='?',
                        help='Path to the language model binary file')
    parser.add_argument('--trie', nargs='?',
                        help='Path to the language model trie file created with native_client/generate_trie')
    parser.add_argument('--audio',
                        help='Path to the audio file to run (WAV format)')
    parser.add_argument('--version',
                        help='Print version and exits')
    args = parser.parse_args()

    if args.version:
        print_versions()
        return 0

    print('Loading model from file {}'.format(args.model), file=sys.stderr)
    model_load_start = timer()
    ds = Model(args.model, N_FEATURES, N_CONTEXT, args.alphabet, BEAM_WIDTH)
    model_load_end = timer() - model_load_start
    print('Loaded model in {:.3}s.'.format(model_load_end), file=sys.stderr)

    if args.lm and args.trie:
        print('Loading language model from files {} {}'.format(args.lm, args.trie), file=sys.stderr)
        lm_load_start = timer()
        ds.enableDecoderWithLM(args.alphabet, args.lm, args.trie, LM_WEIGHT,
                               WORD_COUNT_WEIGHT, VALID_WORD_COUNT_WEIGHT)
        lm_load_end = timer() - lm_load_start
        print('Loaded language model in {:.3}s.'.format(lm_load_end), file=sys.stderr)

    fin = wave.open(args.audio, 'rb')
    fs = fin.getframerate()
    if fs != 16000:
        print('Warning: original sample rate ({}) is different than 16kHz. Resampling might produce erratic speech recognition.'.format(fs), file=sys.stderr)
        fs, audio = convert_samplerate(args.audio)
    else:
        audio = np.frombuffer(fin.readframes(fin.getnframes()), np.int16)

    audio_length = fin.getnframes() * (1/16000)
    fin.close()

    print('Running inference.', file=sys.stderr)
    inference_start = timer()
    print(ds.stt(audio, fs))
    inference_end = timer() - inference_start
    print('Inference took %0.3fs for %0.3fs audio file.' % (inference_end, audio_length), file=sys.stderr)

if __name__ == '__main__':
    main()
