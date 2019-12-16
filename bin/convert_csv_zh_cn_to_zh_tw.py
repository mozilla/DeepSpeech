#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

# Make sure we can import stuff from util/
# This script needs to be run from the root of the DeepSpeech repository

import argparse
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

try:
    from opencc import OpenCC
except ImportError as err:
    print('\n#### [ImportError] try `pip install opencc_python_reimplemented` ####\n')
    raise err

from util.feeding import read_csvs


def _convert_transcript_from_file(from_path, to_path, reverse=False):
    assert from_path != to_path, "don't overwrite CSV to `from file`: {}".format(
        from_path)

    from_csv = read_csvs([from_path])
    assert 'transcript' in from_csv, 'transcript column not found in {}'.format(
        from_path)

    from_lang = 'TW' if reverse else 'CN'
    to_lang = 'CN' if reverse else 'TW'
    conversion = 'tw2s' if reverse else 's2tw'

    print('[Convert] ({}) {}  -->  ({}) {} ...'.format(
        from_lang, from_path, to_lang, to_path
    ))

    cc = OpenCC(conversion)
    from_csv['transcript'] = from_csv['transcript'].apply(cc.convert)

    to_dir = os.path.dirname(to_path)
    if not os.path.exists(to_dir):
        os.makedirs(to_dir)
    from_csv.to_csv(to_path, index=None)


def _convert_transcript_from_dir(from_dir, to_dir, reverse=False):
    assert from_dir != to_dir, \
        "don't overwrite CSVs to same directory: {}".format(from_dir)
    assert os.path.exists(from_dir)

    filenames = os.listdir(from_dir)
    for filename in filenames:
        from_path = os.path.join(from_dir, filename)
        if os.path.isdir(from_path):
            continue
        if not from_path.endswith('.csv'):
            continue

        to_path = os.path.join(to_dir, filename)
        _convert_transcript_from_file(from_path, to_path, reverse)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
        description='Convert `zh-cn` CSV transcript to `zh-tw` CSV transcript')
    PARSER.add_argument('--from_file', help='Convert From CSV File')
    PARSER.add_argument('--to_file', help='Convert To CSV File')
    PARSER.add_argument(
        '--from_dir', help='Convert From directory where contains .csv files')
    PARSER.add_argument('--to_dir', help='Convert .csv Files To directory')
    PARSER.add_argument('--reverse', action='store_true',
                        help='True means convert `zh-tw` to `zh-cn` transcript')
    PARAMS = PARSER.parse_args()

    if PARAMS.from_file and PARAMS.to_file:
        _convert_transcript_from_file(
            PARAMS.from_file, PARAMS.to_file, PARAMS.reverse)

    elif PARAMS.from_dir and PARAMS.to_dir:
        _convert_transcript_from_dir(
            PARAMS.from_dir, PARAMS.to_dir, PARAMS.reverse)
    else:
        print('#### [Argument Error] must specify `--from_file to --to_file` or `--from_dir to --to_dir` ####')
        PARSER.print_help()
