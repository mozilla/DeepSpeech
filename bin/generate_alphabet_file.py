#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

# Make sure we can import stuff from util/
# This script needs to be run from the root of the DeepSpeech repository
import pandas as pd
import argparse
import sys
import os
import re
import unicodedata
from string import punctuation
sys.path.insert(1, os.path.join(sys.path[0], '..'))


class CharCounter():
    def __init__(self, exclude_chars='', include_chars='', normalize=True, no_punctuation=True):
        self._exclude_chars = exclude_chars
        self._include_chars = include_chars
        self._count_map = {}
        self._total_chars = 0
        self._normalize = normalize
        self._no_punctuation = no_punctuation

        removes = r"[{}]".format(
            punctuation +
            r"˙‥‧‵‵❛❜❝❞、。〃〝〞︰︰﹐﹒﹔﹔﹕！＃＄％＆＊，．：；？＠～•…¿" +
            r"“‘·′”’ˊˋˇ—＋｜＼\\　" +
            r"〈〈〉《》「」『』【】〔〕︵︶︷︸︹︺︻︼︽︽︾︿﹀﹁﹁﹂﹃﹄" +
            r"﹙﹙﹚﹛﹜﹝﹞﹤﹥（）＜＞｛｛｝／＝∧∠°÷≡≥≤≠✚∮≦≧✕✖╳≋×±")
        self._punctuation_pattern = re.compile(removes)

    def add_transcript(self, transcript):
        if self._normalize:
            transcript = unicodedata.normalize("NFKD", transcript)
        if self._no_punctuation:
            transcript = self._punctuation_pattern.sub('', transcript)

        for char in transcript.lower():
            try:
                self._count_map[char] += 1
            except KeyError:
                self._count_map[char] = 1
            self._total_chars += 1

    def total_chars(self):
        return self._total_chars

    def char_rate(self, char):
        return float(self._count_map[char]) / float(self._total_chars)

    def export_alphabet(self, output_path, exclude_rate=0.0, exclude_count=0):
        for char in self._include_chars:
            self._count_map[char] = self.total_chars()
        if exclude_rate > 0.0:
            exclude_count = 0
            chars = self._get_chars_by_rate(exclude_rate)
        elif exclude_count > 0:
            exclude_rate = 0
            chars = self._get_chars_by_count(exclude_count)
        else:
            chars = self._get_chars_by_count(0)
        if self._exclude_chars:
            chars = re.sub(r'[{}]'.format(self._exclude_chars), '', chars)

        out_dir = os.path.dirname(output_path)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        with open(output_path, encoding='utf8', mode='w+') as thefile:
            thefile.write(
                '#   This Config File is Auto Generated with Info below:\n')
            thefile.write(
                '#  [Total Characters Count] \t{}\n'.format(self._total_chars))
            thefile.write(
                '#   [Total Alphabet Count]  \t{}\n'.format(len(self._count_map)))
            thefile.write(
                '#   [Valid Alphabet Count]  \t{}\n'.format(len(chars)))
            thefile.write(
                '#      [--exclude_chars]     \t"{}"\n'.format(self._exclude_chars))
            thefile.write(
                '#      [--include_chars]     \t"{}"\n'.format(self._include_chars))
            if exclude_count != 0:
                thefile.write(
                    '#   [--exclude_chars_count]  \t{}\n'.format(exclude_count))
            if exclude_rate != 0.0:
                thefile.write(
                    '#   [--exclude_chars_rate]  \t{}\n'.format(exclude_rate))
            thefile.write(
                '#      [--no_normalize]     \t{}\n'.format(not self._normalize))
            thefile.write(
                '#   [--include_punctuation]  \t{}\n'.format(not self._no_punctuation))

            chars = ''.join(sorted(chars))
            for char in chars:
                thefile.write(char + '\n')
            thefile.write('# end of the file\n')

    def _get_chars_by_rate(self, rate_baseline):
        chars = []
        for char in self._count_map:
            if self.char_rate(char) > rate_baseline:
                chars.append(char)
        return ''.join(chars)

    def _get_chars_by_count(self, count_baseline):
        chars = []
        for char, count in self._count_map.items():
            if count > count_baseline:
                chars.append(char)
        return ''.join(chars)

    def walk_dir(self, dirname):
        for dirpath, _, filenames in os.walk(dirname):
            for filename in filenames:
                if filename.endswith('.csv'):
                    filepath = os.path.join(dirpath, filename)
                    df = pd.read_csv(filepath)
                    if 'transcript' not in df:
                        print(
                            '[bypass] ignore csv without transcript column: {}'.format(filepath))
                        continue

                    print('collect transcript from {}'.format(filepath))
                    for transcript in df['transcript']:
                        self.add_transcript(transcript)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
        description='Generate Alphabet File from CSV Files')
    PARSER.add_argument(
        'walk_dir', help='Walk Directory for every csv with valid format (`transcript` column)')
    PARSER.add_argument('output', help='Output Filename of Alphabet')
    PARSER.add_argument('--exclude_chars',
                        help='Never Include Specified Characters as Alphabet')
    PARSER.add_argument('--include_chars',
                        help='Must Include Specified Characters as Alphabet')
    PARSER.add_argument('--exclude_chars_rate',
                        help='specify 0.0 < [float] < 1.0 to exclude characters which occurrence rate lower than --include_chars_rate')
    PARSER.add_argument('--exclude_chars_count',
                        help='specify 0 < [int] to exclude characters which occurrence count lower than --include_chars_count')
    PARSER.add_argument('--no_normalize', action='store_true',
                        help="Don't normalize characters, default is `NORMALIZE`")
    PARSER.add_argument('--include_punctuation', action='store_true',
                        help="Don't Exclude Punctuation, default is `EXCLUDE`")

    PARAMS = PARSER.parse_args()

    PARAMS.exclude_chars = PARAMS.exclude_chars if PARAMS.exclude_chars else ''
    PARAMS.include_chars = PARAMS.include_chars if PARAMS.include_chars else ''
    try:
        PARAMS.exclude_chars_rate = float(PARAMS.exclude_chars_rate)
    except TypeError:
        PARAMS.exclude_chars_rate = 0.0

    try:
        PARAMS.exclude_chars_count = int(PARAMS.exclude_chars_count)
    except TypeError:
        PARAMS.exclude_chars_count = 0

    PARAMS.no_normalize = bool(PARAMS.no_normalize)
    PARAMS.include_punctuation = bool(PARAMS.include_punctuation)
    CHAR_COUNTER = CharCounter(
        exclude_chars=PARAMS.exclude_chars,
        include_chars=PARAMS.include_chars,
        normalize=not PARAMS.no_normalize,
        no_punctuation=not PARAMS.include_punctuation)
    CHAR_COUNTER.walk_dir(PARAMS.walk_dir)
    CHAR_COUNTER.export_alphabet(
        PARAMS.output, PARAMS.exclude_chars_rate, PARAMS.exclude_chars_count)
