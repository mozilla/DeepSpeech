#!/usr/bin/env python
"""
Tool for building Sample Databases (SDB files) from DeepSpeech CSV files and other SDB files
Use "python3 build_sdb.py -h" for help
"""
import argparse

import progressbar

from deepspeech_training.util.audio import (
    AUDIO_TYPE_PCM,
    AUDIO_TYPE_OPUS,
    AUDIO_TYPE_WAV,
    change_audio_types,
)
from deepspeech_training.util.downloader import SIMPLE_BAR
from deepspeech_training.util.sample_collections import (
    DirectSDBWriter,
    samples_from_sources,
)
from deepspeech_training.util.augmentations import (
    parse_augmentations,
    apply_sample_augmentations,
    SampleAugmentation
)

AUDIO_TYPE_LOOKUP = {"wav": AUDIO_TYPE_WAV, "opus": AUDIO_TYPE_OPUS}


def build_sdb():
    audio_type = AUDIO_TYPE_LOOKUP[CLI_ARGS.audio_type]
    augmentations = parse_augmentations(CLI_ARGS.augment)
    if any(not isinstance(a, SampleAugmentation) for a in augmentations):
        print("Warning: Some of the augmentations cannot be applied by this command.")
    with DirectSDBWriter(
        CLI_ARGS.target, audio_type=audio_type, labeled=not CLI_ARGS.unlabeled
    ) as sdb_writer:
        samples = samples_from_sources(CLI_ARGS.sources, labeled=not CLI_ARGS.unlabeled)
        num_samples = len(samples)
        if augmentations:
            samples = apply_sample_augmentations(samples, audio_type=AUDIO_TYPE_PCM, augmentations=augmentations)
        bar = progressbar.ProgressBar(max_value=num_samples, widgets=SIMPLE_BAR)
        for sample in bar(
            change_audio_types(samples, audio_type=audio_type, bitrate=CLI_ARGS.bitrate, processes=CLI_ARGS.workers)
        ):
            sdb_writer.add(sample)


def handle_args():
    parser = argparse.ArgumentParser(
        description="Tool for building Sample Databases (SDB files) "
        "from DeepSpeech CSV files and other SDB files"
    )
    parser.add_argument(
        "sources",
        nargs="+",
        help="Source CSV and/or SDB files - "
        "Note: For getting a correctly ordered target SDB, source SDBs have to have their samples "
        "already ordered from shortest to longest.",
    )
    parser.add_argument("target", help="SDB file to create")
    parser.add_argument(
        "--audio-type",
        default="opus",
        choices=AUDIO_TYPE_LOOKUP.keys(),
        help="Audio representation inside target SDB",
    )
    parser.add_argument(
        "--bitrate",
        type=int,
        help="Bitrate for lossy compressed SDB samples like in case of --audio-type opus",
    )
    parser.add_argument(
        "--workers", type=int, default=None, help="Number of encoding SDB workers"
    )
    parser.add_argument(
        "--unlabeled",
        action="store_true",
        help="If to build an SDB with unlabeled (audio only) samples - "
        "typically used for building noise augmentation corpora",
    )
    parser.add_argument(
        "--augment",
        action='append',
        help="Add an augmentation operation",
    )
    return parser.parse_args()


if __name__ == "__main__":
    CLI_ARGS = handle_args()
    build_sdb()
