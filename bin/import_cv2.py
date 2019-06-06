#!/usr/bin/env python
'''
Broadly speaking, this script takes the audio downloaded from Common Voice
for a certain language, in addition to the *.tsv files output by CorporaCreator,
and the script formats the data and transcripts to be in a state usable by
DeepSpeech.py
Use "python3 import_cv2.py -h" for help
'''
from __future__ import absolute_import, division, print_function

# Make sure we can import stuff from util/
# This script needs to be run from the root of the DeepSpeech repository
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import csv
import sox
import argparse
import subprocess
import progressbar
import unicodedata

from os import path
from threading import RLock
from multiprocessing.dummy import Pool
from multiprocessing import cpu_count
from util.downloader import SIMPLE_BAR
from util.text import Alphabet, validate_label
from util.feeding import secs_to_hours


FIELDNAMES = ['wav_filename', 'wav_filesize', 'transcript']
SAMPLE_RATE = 16000
MAX_SECS = 10


def _preprocess_data(tsv_dir, audio_dir, label_filter, space_after_every_character=False):
    for dataset in ['train', 'test', 'dev']:
        input_tsv = path.join(path.abspath(tsv_dir), dataset+".tsv")
        if os.path.isfile(input_tsv):
            print("Loading TSV file: ", input_tsv)
            _maybe_convert_set(input_tsv, audio_dir, label_filter, space_after_every_character)
        else:
            print("ERROR: no TSV file found: ", input_tsv)


def _maybe_convert_set(input_tsv, audio_dir, label_filter, space_after_every_character=None):
    output_csv = path.join(audio_dir, os.path.split(input_tsv)[-1].replace('tsv', 'csv'))
    print("Saving new DeepSpeech-formatted CSV file to: ", output_csv)

    # Get audiofile path and transcript for each sentence in tsv
    samples = []
    with open(input_tsv) as input_tsv_file:
        reader = csv.DictReader(input_tsv_file, delimiter='\t')
        for row in reader:
            samples.append((row['path'], row['sentence']))

    # Keep track of how many samples are good vs. problematic
    counter = {'all': 0, 'failed': 0, 'invalid_label': 0, 'too_short': 0, 'too_long': 0, 'total_time': 0}
    lock = RLock()
    num_samples = len(samples)
    rows = []

    def one_sample(sample):
        """ Take a audio file, and optionally convert it to 16kHz WAV """
        mp3_filename = path.join(audio_dir, sample[0])
        if not path.splitext(mp3_filename.lower())[1] == '.mp3':
            mp3_filename += ".mp3"
        # Storing wav files next to the mp3 ones - just with a different suffix
        wav_filename = path.splitext(mp3_filename)[0] + ".wav"
        _maybe_convert_wav(mp3_filename, wav_filename)
        file_size = -1
        frames = 0
        if path.exists(wav_filename):
            file_size = path.getsize(wav_filename)
            frames = int(subprocess.check_output(['soxi', '-s', wav_filename], stderr=subprocess.STDOUT))
        label = label_filter(sample[1])
        with lock:
            if file_size == -1:
                # Excluding samples that failed upon conversion
                counter['failed'] += 1
            elif label is None:
                # Excluding samples that failed on label validation
                counter['invalid_label'] += 1
            elif int(frames/SAMPLE_RATE*1000/10/2) < len(str(label)):
                # Excluding samples that are too short to fit the transcript
                counter['too_short'] += 1
            elif frames/SAMPLE_RATE > MAX_SECS:
                # Excluding very long samples to keep a reasonable batch-size
                counter['too_long'] += 1
            else:
                # This one is good - keep it for the target CSV
                rows.append((wav_filename, file_size, label))
            counter['all'] += 1
            counter['total_time'] += frames

    print("Importing mp3 files...")
    pool = Pool(cpu_count())
    bar = progressbar.ProgressBar(max_value=num_samples, widgets=SIMPLE_BAR)
    for i, _ in enumerate(pool.imap_unordered(one_sample, samples), start=1):
        bar.update(i)
    bar.update(num_samples)
    pool.close()
    pool.join()

    with open(output_csv, 'w') as output_csv_file:
        print('Writing CSV file for DeepSpeech.py as: ', output_csv)
        writer = csv.DictWriter(output_csv_file, fieldnames=FIELDNAMES)
        writer.writeheader()
        bar = progressbar.ProgressBar(max_value=len(rows), widgets=SIMPLE_BAR)
        for filename, file_size, transcript in bar(rows):
            if space_after_every_character:
                writer.writerow({'wav_filename': filename, 'wav_filesize': file_size, 'transcript': ' '.join(transcript)})
            else:
                writer.writerow({'wav_filename': filename, 'wav_filesize': file_size, 'transcript': transcript})

    print('Imported %d samples.' % (counter['all'] - counter['failed'] - counter['too_short'] - counter['too_long']))
    if counter['failed'] > 0:
        print('Skipped %d samples that failed upon conversion.' % counter['failed'])
    if counter['invalid_label'] > 0:
        print('Skipped %d samples that failed on transcript validation.' % counter['invalid_label'])
    if counter['too_short'] > 0:
        print('Skipped %d samples that were too short to match the transcript.' % counter['too_short'])
    if counter['too_long'] > 0:
        print('Skipped %d samples that were longer than %d seconds.' % (counter['too_long'], MAX_SECS))
    print('Final amount of imported audio: %s.' % secs_to_hours(counter['total_time'] / SAMPLE_RATE))


def _maybe_convert_wav(mp3_filename, wav_filename):
    if not path.exists(wav_filename):
        transformer = sox.Transformer()
        transformer.convert(samplerate=SAMPLE_RATE)
        try:
            transformer.build(mp3_filename, wav_filename)
        except sox.core.SoxError:
            pass


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description='Import CommonVoice v2.0 corpora')
    PARSER.add_argument('tsv_dir', help='Directory containing tsv files')
    PARSER.add_argument('--audio_dir', help='Directory containing the audio clips - defaults to "<tsv_dir>/clips"')
    PARSER.add_argument('--filter_alphabet', help='Exclude samples with characters not in provided alphabet')
    PARSER.add_argument('--normalize', action='store_true', help='Converts diacritic characters to their base ones')
    PARSER.add_argument('--space_after_every_character', action='store_true', help='To help transcript join by white space')

    PARAMS = PARSER.parse_args()

    AUDIO_DIR = PARAMS.audio_dir if PARAMS.audio_dir else os.path.join(PARAMS.tsv_dir, 'clips')
    ALPHABET = Alphabet(PARAMS.filter_alphabet) if PARAMS.filter_alphabet else None

    def label_filter_fun(label):
        if PARAMS.normalize:
            label = unicodedata.normalize("NFKD", label.strip()) \
                .encode("ascii", "ignore") \
                .decode("ascii", "ignore")
        label = validate_label(label)
        if ALPHABET and label:
            try:
                [ALPHABET.label_from_string(c) for c in label]
            except KeyError:
                label = None
        return label

    _preprocess_data(PARAMS.tsv_dir, AUDIO_DIR, label_filter_fun, PARAMS.space_after_every_character)
