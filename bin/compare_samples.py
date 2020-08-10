#!/usr/bin/env python
"""
Tool for comparing two wav samples
"""
import sys
import argparse

from mozilla_voice_stt_training.util.audio import AUDIO_TYPE_NP, mean_dbfs
from mozilla_voice_stt_training.util.sample_collections import load_sample


def fail(message):
    print(message, file=sys.stderr, flush=True)
    sys.exit(1)


def compare_samples():
    sample1 = load_sample(CLI_ARGS.sample1)
    sample2 = load_sample(CLI_ARGS.sample2)
    if sample1.audio_format != sample2.audio_format:
        fail('Samples differ on: audio-format ({} and {})'.format(sample1.audio_format, sample2.audio_format))
    if sample1.duration != sample2.duration:
        fail('Samples differ on: duration ({} and {})'.format(sample1.duration, sample2.duration))
    sample1.change_audio_type(AUDIO_TYPE_NP)
    sample2.change_audio_type(AUDIO_TYPE_NP)
    audio_diff = sample1.audio - sample2.audio
    diff_dbfs = mean_dbfs(audio_diff)
    differ_msg = 'Samples differ on: sample data ({:0.2f} dB difference) '.format(diff_dbfs)
    equal_msg = 'Samples are considered equal ({:0.2f} dB difference)'.format(diff_dbfs)
    if CLI_ARGS.if_differ:
        if diff_dbfs <= CLI_ARGS.threshold:
            fail(equal_msg)
        if not CLI_ARGS.no_success_output:
            print(differ_msg, file=sys.stderr, flush=True)
    else:
        if diff_dbfs > CLI_ARGS.threshold:
            fail(differ_msg)
        if not CLI_ARGS.no_success_output:
            print(equal_msg, file=sys.stderr, flush=True)


def handle_args():
    parser = argparse.ArgumentParser(
        description="Tool for checking similarity of two samples"
    )
    parser.add_argument("sample1", help="Filename of sample 1 to compare")
    parser.add_argument("sample2", help="Filename of sample 2 to compare")
    parser.add_argument("--threshold", type=float, default=-60.0,
                        help="dB of sample deltas above which they are considered different")
    parser.add_argument(
        "--if-differ",
        action="store_true",
        help="If to succeed and return status code 0 on different signals and fail on equal ones (inverse check)."
             "This will still fail on different formats or durations.",
    )
    parser.add_argument(
        "--no-success-output",
        action="store_true",
        help="Stay silent on success (if samples are equal of - with --if-differ - samples are not equal)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    CLI_ARGS = handle_args()
    compare_samples()
