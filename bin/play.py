#!/usr/bin/env python
"""
Tool for playing (and augmenting) single samples or samples from Sample Databases (SDB files) and DeepSpeech CSV files
Use "python3 play.py -h" for help
"""

import os
import sys
import random
import argparse

from mozilla_voice_stt_training.util.audio import LOADABLE_AUDIO_EXTENSIONS, AUDIO_TYPE_PCM, AUDIO_TYPE_WAV
from mozilla_voice_stt_training.util.sample_collections import SampleList, LabeledSample, samples_from_source
from mozilla_voice_stt_training.util.augmentations import parse_augmentations, apply_sample_augmentations, SampleAugmentation


def get_samples_in_play_order():
    ext = os.path.splitext(CLI_ARGS.source)[1].lower()
    if ext in LOADABLE_AUDIO_EXTENSIONS:
        samples = SampleList([(CLI_ARGS.source, 0)], labeled=False)
    else:
        samples = samples_from_source(CLI_ARGS.source, buffering=0)
    played = 0
    index = CLI_ARGS.start
    while True:
        if 0 <= CLI_ARGS.number <= played:
            return
        if CLI_ARGS.random:
            yield samples[random.randint(0, len(samples) - 1)]
        elif index < 0:
            yield samples[len(samples) + index]
        elif index >= len(samples):
            print("No sample with index {}".format(CLI_ARGS.start))
            sys.exit(1)
        else:
            yield samples[index]
        played += 1
        index = (index + 1) % len(samples)


def play_collection():
    augmentations = parse_augmentations(CLI_ARGS.augment)
    if any(not isinstance(a, SampleAugmentation) for a in augmentations):
        print("Warning: Some of the augmentations cannot be simulated by this command.")
    samples = get_samples_in_play_order()
    samples = apply_sample_augmentations(samples,
                                         audio_type=AUDIO_TYPE_PCM,
                                         augmentations=augmentations,
                                         process_ahead=0,
                                         clock=CLI_ARGS.clock)
    for sample in samples:
        if not CLI_ARGS.quiet:
            print('Sample "{}"'.format(sample.sample_id), file=sys.stderr)
            if isinstance(sample, LabeledSample):
                print('  "{}"'.format(sample.transcript), file=sys.stderr)
        if CLI_ARGS.pipe:
            sample.change_audio_type(AUDIO_TYPE_WAV)
            sys.stdout.buffer.write(sample.audio.getvalue())
            return
        wave_obj = simpleaudio.WaveObject(sample.audio,
                                          sample.audio_format.channels,
                                          sample.audio_format.width,
                                          sample.audio_format.rate)
        play_obj = wave_obj.play()
        play_obj.wait_done()


def handle_args():
    parser = argparse.ArgumentParser(
        description="Tool for playing (and augmenting) single samples or samples from Sample Databases (SDB files) "
        "and DeepSpeech CSV files"
    )
    parser.add_argument("source", help="Sample DB, CSV or WAV file to play samples from")
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Sample index to start at (negative numbers are relative to the end of the collection)",
    )
    parser.add_argument(
        "--number",
        type=int,
        default=-1,
        help="Number of samples to play (-1 for endless)",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="If samples should be played in random order",
    )
    parser.add_argument(
        "--augment",
        action='append',
        help="Add an augmentation operation",
    )
    parser.add_argument(
        "--clock",
        type=float,
        default=0.5,
        help="Simulates clock value used for augmentations during training."
             "Ranges from 0.0 (representing parameter start values) to"
             "1.0 (representing parameter end values)",
    )
    parser.add_argument(
        "--pipe",
        action="store_true",
        help="Pipe first sample as wav file to stdout. Forces --number to 1.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="No info logging to console",
    )
    return parser.parse_args()


if __name__ == "__main__":
    CLI_ARGS = handle_args()
    if not CLI_ARGS.pipe:
        try:
            import simpleaudio
        except ModuleNotFoundError:
            print('Unless using the --pipe flag, play.py requires Python package "simpleaudio" for playing samples')
            sys.exit(1)
    try:
        play_collection()
    except KeyboardInterrupt:
        print(" Stopped")
        sys.exit(0)
