#!/usr/bin/env python
'''
Tool for building Sample Databases (SDB files) from DeepSpeech CSV files and other SDB files
Use "python3 build_sdb.py -h" for help
'''
from __future__ import absolute_import, division, print_function

# Make sure we can import stuff from util/
# This script needs to be run from the root of the DeepSpeech repository
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import argparse
import progressbar

from util.downloader import SIMPLE_BAR
from util.audio import change_audio_types, AUDIO_TYPE_WAV, AUDIO_TYPE_OPUS
from util.sample_collections import samples_from_files, DirectSDBWriter

AUDIO_TYPE_LOOKUP = {
    'wav': AUDIO_TYPE_WAV,
    'opus': AUDIO_TYPE_OPUS
}


def build_sdb():
    audio_type = AUDIO_TYPE_LOOKUP[CLI_ARGS.audio_type]
    with DirectSDBWriter(CLI_ARGS.target, audio_type=audio_type) as sdb_writer:
        samples = samples_from_files(CLI_ARGS.sources)
        bar = progressbar.ProgressBar(max_value=len(samples), widgets=SIMPLE_BAR)
        for sample in bar(change_audio_types(samples, audio_type=audio_type, processes=CLI_ARGS.workers)):
            sdb_writer.add(sample)


def handle_args():
    parser = argparse.ArgumentParser(description='Tool for building Sample Databases (SDB files) '
                                                 'from DeepSpeech CSV files and other SDB files')
    parser.add_argument('sources', nargs='+', help='Source CSV and/or SDB files - '
                                                   'Note: For getting a correctly ordered target SDB, source SDBs have '
                                                   'to have their samples already ordered from shortest to longest.')
    parser.add_argument('target', help='SDB file to create')
    parser.add_argument('--audio-type', default='opus', choices=AUDIO_TYPE_LOOKUP.keys(),
                        help='Audio representation inside target SDB')
    parser.add_argument('--workers', type=int, default=None, help='Number of encoding SDB workers')
    return parser.parse_args()


if __name__ == "__main__":
    CLI_ARGS = handle_args()
    build_sdb()
