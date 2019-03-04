from __future__ import absolute_import, division, print_function

import codecs
import numpy as np
import re
import sys

from six.moves import range

class Alphabet(object):
    def __init__(self, config_file):
        self._config_file = config_file
        self._size = 256

    def string_from_label(self, label):
        assert False

    def label_from_string(self, string):
        assert False

    def decode(self, labels):
        out=bytes(labels).decode('utf-8')
        # try:
        #     out=bytes(labels).decode('utf-8')
        # except UnicodeDecodeError:
        #     out=bytes(labels)
        return out

    def size(self):
        return self._size

    def config_file(self):
        return self._config_file


def text_to_char_array(original, alphabet):
    r"""
    Given a Python string ``original``, remove unsupported characters, map characters
    to integers and return a numpy array representing the processed string.
    """
    return np.asarray([alphabet.label_from_string(c) for c in original])


def wer_cer_batch(originals, results):
    r"""
    The WER is defined as the editing/Levenshtein distance on word level
    divided by the amount of words in the original text.
    In case of the original having more words (N) than the result and both
    being totally different (all N words resulting in 1 edit operation each),
    the WER will always be 1 (N / N = 1).
    """
    # The WER is calculated on word (and NOT on character) level.
    # Therefore we split the strings into words first
    assert len(originals) == len(results)

    total_cer = 0.0
    total_char_length = 0.0

    total_wer = 0.0
    total_word_length = 0.0

    for original, result in zip(originals, results):
        total_cer += levenshtein(original, result)
        total_char_length += len(original)

        total_wer += levenshtein(original.split(), result.split())
        total_word_length += len(original.split())

    return total_wer / total_word_length, total_cer / total_char_length


# The following code is from: http://hetland.org/coding/python/levenshtein.py

# This is a straightforward implementation of a well-known algorithm, and thus
# probably shouldn't be covered by copyright to begin with. But in case it is,
# the author (Magnus Lie Hetland) has, to the extent possible under law,
# dedicated all copyright and related and neighboring rights to this software
# to the public domain worldwide, by distributing it under the CC0 license,
# version 1.0. This software is distributed without any warranty. For more
# information, see <http://creativecommons.org/publicdomain/zero/1.0>

def levenshtein(a,b):
    "Calculates the Levenshtein distance between a and b."
    n, m = len(a), len(b)
    if n > m:
        # Make sure n <= m, to use O(min(n,m)) space
        a,b = b,a
        n,m = m,n

    current = list(range(n+1))
    for i in range(1,m+1):
        previous, current = current, [i]+[0]*n
        for j in range(1,n+1):
            add, delete = previous[j]+1, current[j-1]+1
            change = previous[j-1]
            if a[j-1] != b[i-1]:
                change = change + 1
            current[j] = min(add, delete, change)

    return current[n]

# Validate and normalize transcriptions. Returns a cleaned version of the label
# or None if it's invalid.
def validate_label(label):
    # For now we can only handle [a-z ']
    if "(" in label or \
                    "<" in label or \
                    "[" in label or \
                    "]" in label or \
                    "&" in label or \
                    "*" in label or \
                    "{" in label or \
            re.search(r"[0-9]", label) != None:
        return None

    label = label.replace("-", "")
    label = label.replace("_", "")
    label = label.replace(".", "")
    label = label.replace(",", "")
    label = label.replace("?", "")
    label = label.strip()

    return label.lower()
