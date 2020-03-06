from __future__ import absolute_import, division, print_function

import numpy as np
import re
import struct

from util.flags import FLAGS
from six.moves import range

class Alphabet(object):
    def __init__(self, config_file):
        self._config_file = config_file
        self._label_to_str = {}
        self._str_to_label = {}
        self._size = 0
        if config_file:
            with open(config_file, 'r', encoding='utf-8') as fin:
                for line in fin:
                    if line[0:2] == '\\#':
                        line = '#\n'
                    elif line[0] == '#':
                        continue
                    self._label_to_str[self._size] = line[:-1] # remove the line ending
                    self._str_to_label[line[:-1]] = self._size
                    self._size += 1

    def _string_from_label(self, label):
        return self._label_to_str[label]

    def _label_from_string(self, string):
        try:
            return self._str_to_label[string]
        except KeyError as e:
            raise KeyError(
                'ERROR: Your transcripts contain characters (e.g. \'{}\') which do not occur in \'{}\'! Use ' \
                'util/check_characters.py to see what characters are in your [train,dev,test].csv transcripts, and ' \
                'then add all these to \'{}\'.'.format(string, self._config_file, self._config_file)
            ).with_traceback(e.__traceback__)

    def has_char(self, char):
        return char in self._str_to_label

    def encode(self, string):
        res = []
        for char in string:
            res.append(self._label_from_string(char))
        return res

    def decode(self, labels):
        res = ''
        for label in labels:
            res += self._string_from_label(label)
        return res

    def serialize(self):
        # Serialization format is a sequence of (key, value) pairs, where key is
        # a uint16_t and value is a uint16_t length followed by `length` UTF-8
        # encoded bytes with the label.
        res = bytearray()

        # We start by writing the number of pairs in the buffer as uint16_t.
        res += struct.pack('<H', self._size)
        for key, value in self._label_to_str.items():
            value = value.encode('utf-8')
            # struct.pack only takes fixed length strings/buffers, so we have to
            # construct the correct format string with the length of the encoded
            # label.
            res += struct.pack('<HH{}s'.format(len(value)), key, len(value), value)
        return bytes(res)

    def size(self):
        return self._size

    def config_file(self):
        return self._config_file


class UTF8Alphabet(object):
    @staticmethod
    def _string_from_label(_):
        assert False

    @staticmethod
    def _label_from_string(_):
        assert False

    @staticmethod
    def encode(string):
        # 0 never happens in the data, so we can shift values by one, use 255 for
        # the CTC blank, and keep the alphabet size = 256
        return np.frombuffer(string.encode('utf-8'), np.uint8).astype(np.int32) - 1

    @staticmethod
    def decode(labels):
        # And here we need to shift back up
        return bytes(np.asarray(labels, np.uint8) + 1).decode('utf-8', errors='replace')

    @staticmethod
    def size():
        return 255

    @staticmethod
    def serialize():
        res = bytearray()
        res += struct.pack('<h', 255)
        for i in range(255):
            # Note that we also shift back up in the mapping constructed here
            # so that the native client sees the correct byte values when decoding.
            res += struct.pack('<hh1s', i, 1, bytes([i+1]))
        return bytes(res)

    @staticmethod
    def deserialize(buf):
        size = struct.unpack('<I', buf)[0]
        assert size == 255
        return UTF8Alphabet()

    @staticmethod
    def config_file():
        return ''


def text_to_char_array(series, alphabet):
    r"""
    Given a Pandas Series containing transcript string, map characters to
    integers and return a numpy array representing the processed string.
    """
    try:
        transcript = np.asarray(alphabet.encode(series['transcript']))
        if len(transcript) == 0:
            raise ValueError('While processing: {}\nFound an empty transcript! You must include a transcript for all training data.'.format(series['wav_filename']))
        return transcript
    except KeyError as e:
        # Provide the row context (especially wav_filename) for alphabet errors
        raise ValueError('While processing: {}\n{}'.format(series['wav_filename'], e))


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

# Validate and normalize transcriptions. Returns a cleaned version of the label
# or None if it's invalid.
def validate_label(label):
    # For now we can only handle [a-z ']
    if re.search(r"[0-9]|[(<\[\]&*{]", label) is not None:
        return None

    label = label.replace("-", " ")
    label = label.replace("_", " ")
    label = re.sub("[ ]{2,}", " ", label)
    label = label.replace(".", "")
    label = label.replace(",", "")
    label = label.replace(";", "")
    label = label.replace("?", "")
    label = label.replace("!", "")
    label = label.replace(":", "")
    label = label.replace("\"", "")
    label = label.strip()
    label = label.lower()

    return label if label else None
