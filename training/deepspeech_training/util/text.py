from __future__ import absolute_import, division, print_function

import numpy as np
import struct

def text_to_char_array(transcript, alphabet, context=''):
    r"""
    Given a transcript string, map characters to
    integers and return a numpy array representing the processed string.
    Use a string in `context` for adding text to raised exceptions.
    """
    if not alphabet.CanEncode(transcript):
        # Provide the row context (especially wav_filename) for alphabet errors
        raise ValueError(
            'Alphabet cannot encode transcript "{}" while processing sample "{}", '
            'check that your alphabet contains all characters in the training corpus. '
            'Missing characters are: {}.'
            .format(transcript, context, list(ch for ch in transcript if not alphabet.CanEncodeSingle(ch))))

    encoded = alphabet.Encode(transcript)
    if len(encoded) == 0:
        raise ValueError('While processing {}: Found an empty transcript! '
                         'You must include a transcript for all training data.'
                         .format(context))
    return encoded


# The following code is from: http://hetland.org/coding/python/levenshtein.py

# This is a straightforward implementation of a well-known algorithm, and thus
# probably shouldn't be covered by copyright to begin with. But in case it is,
# the author (Magnus Lie Hetland) has, to the extent possible under law,
# dedicated all copyright and related and neighboring rights to this software
# to the public domain worldwide, by distributing it under the CC0 license,
# version 1.0. This software is distributed without any warranty. For more
# information, see <http://creativecommons.org/publicdomain/zero/1.0>

def levenshtein(a, b):
    "Calculates the Levenshtein distance between a and b."
    n, m = len(a), len(b)
    if n > m:
        # Make sure n <= m, to use O(min(n,m)) space
        a, b = b, a
        n, m = m, n

    current = list(range(n+1))
    for i in range(1, m+1):
        previous, current = current, [i]+[0]*n
        for j in range(1, n+1):
            add, delete = previous[j]+1, current[j-1]+1
            change = previous[j-1]
            if a[j-1] != b[i-1]:
                change = change + 1
            current[j] = min(add, delete, change)

    return current[n]
