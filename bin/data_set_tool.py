#!/usr/bin/env python
'''
Tool for building a combined SDB or CSV sample-set from other sets
Use 'python3 data_set_tool.py -h' for help
'''
import sys
import argparse
import progressbar
from pathlib import Path

from deepspeech_training.util.audio import (
    AUDIO_TYPE_PCM,
    AUDIO_TYPE_OPUS,
    AUDIO_TYPE_WAV,
    change_audio_types,
)
from deepspeech_training.util.downloader import SIMPLE_BAR
from deepspeech_training.util.sample_collections import (
    CSVWriter,
    DirectSDBWriter,
    TarWriter,
    samples_from_sources,
)
from deepspeech_training.util.augmentations import (
    parse_augmentations,
    apply_sample_augmentations,
    SampleAugmentation
)

AUDIO_TYPE_LOOKUP = {'wav': AUDIO_TYPE_WAV, 'opus': AUDIO_TYPE_OPUS}


def build_data_set():
    audio_type = AUDIO_TYPE_LOOKUP[CLI_ARGS.audio_type]
    augmentations = parse_augmentations(CLI_ARGS.augment)
    if any(not isinstance(a, SampleAugmentation) for a in augmentations):
        print('Warning: Some of the specified augmentations will not get applied, as this tool only supports '
              'overlay, codec, reverb, resample and volume.')
    extension = Path(CLI_ARGS.target).suffix.lower()
    labeled = not CLI_ARGS.unlabeled
    if extension == '.csv':
        writer = CSVWriter(CLI_ARGS.target, absolute_paths=CLI_ARGS.absolute_paths, labeled=labeled)
    elif extension == '.sdb':
        writer = DirectSDBWriter(CLI_ARGS.target, audio_type=audio_type, labeled=labeled)
    elif extension == '.tar':
        writer = TarWriter(CLI_ARGS.target, labeled=labeled, gz=False, include=CLI_ARGS.include)
    elif extension == '.tgz' or CLI_ARGS.target.lower().endswith('.tar.gz'):
        writer = TarWriter(CLI_ARGS.target, labeled=labeled, gz=True, include=CLI_ARGS.include)
    else:
        print('Unknown extension of target file - has to be either .csv, .sdb, .tar, .tar.gz or .tgz')
        sys.exit(1)
    with writer:
        samples = samples_from_sources(CLI_ARGS.sources, labeled=not CLI_ARGS.unlabeled)
        num_samples = len(samples)
        if augmentations:
            samples = apply_sample_augmentations(samples, audio_type=AUDIO_TYPE_PCM, augmentations=augmentations)
        bar = progressbar.ProgressBar(max_value=num_samples, widgets=SIMPLE_BAR)
        for sample in bar(change_audio_types(
                samples,
                audio_type=audio_type,
                bitrate=CLI_ARGS.bitrate,
                processes=CLI_ARGS.workers)):
            writer.add(sample)


def handle_args():
    parser = argparse.ArgumentParser(
        description='Tool for building a combined SDB or CSV sample-set from other sets'
    )
    parser.add_argument(
        'sources',
        nargs='+',
        help='Source CSV and/or SDB files - '
        'Note: For getting a correctly ordered target set, source SDBs have to have their samples '
        'already ordered from shortest to longest.',
    )
    parser.add_argument(
        'target',
        help='SDB, CSV or TAR(.gz) file to create'
    )
    parser.add_argument(
        '--audio-type',
        default='opus',
        choices=AUDIO_TYPE_LOOKUP.keys(),
        help='Audio representation inside target SDB',
    )
    parser.add_argument(
        '--bitrate',
        type=int,
        help='Bitrate for lossy compressed SDB samples like in case of --audio-type opus',
    )
    parser.add_argument(
        '--workers', type=int, default=None, help='Number of encoding SDB workers'
    )
    parser.add_argument(
        '--unlabeled',
        action='store_true',
        help='If to build an data-set with unlabeled (audio only) samples - '
        'typically used for building noise augmentation corpora',
    )
    parser.add_argument(
        '--absolute-paths',
        action='store_true',
        help='If to reference samples by their absolute paths when writing CSV files',
    )
    parser.add_argument(
        '--augment',
        action='append',
        help='Add an augmentation operation',
    )
    parser.add_argument(
        '--include',
        action='append',
        help='Adds a file to the root directory of .tar(.gz) targets',
    )
    return parser.parse_args()


if __name__ == '__main__':
    CLI_ARGS = handle_args()
    build_data_set()
