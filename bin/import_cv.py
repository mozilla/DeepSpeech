#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

# Make sure we can import stuff from util/
# This script needs to be run from the root of the DeepSpeech repository
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import csv
import sox
import tarfile
import subprocess
import progressbar

from glob import glob
from os import path
from threading import RLock
from multiprocessing.dummy import Pool
from multiprocessing import cpu_count

from util.downloader import maybe_download, SIMPLE_BAR

FIELDNAMES = ['wav_filename', 'wav_filesize', 'transcript']
SAMPLE_RATE = 16000
MAX_SECS = 10
ARCHIVE_DIR_NAME = 'cv_corpus_v1'
ARCHIVE_NAME = ARCHIVE_DIR_NAME + '.tar.gz'
ARCHIVE_URL = 'https://s3.us-east-2.amazonaws.com/common-voice-data-download/' + ARCHIVE_NAME

def _download_and_preprocess_data(target_dir):
    # Making path absolute
    target_dir = path.abspath(target_dir)
    # Conditionally download data
    archive_path = maybe_download(ARCHIVE_NAME, target_dir, ARCHIVE_URL)
    # Conditionally extract common voice data
    _maybe_extract(target_dir, ARCHIVE_DIR_NAME, archive_path)
    # Conditionally convert common voice CSV files and mp3 data to DeepSpeech CSVs and wav
    _maybe_convert_sets(target_dir, ARCHIVE_DIR_NAME)

def _maybe_extract(target_dir, extracted_data, archive_path):
    # If target_dir/extracted_data does not exist, extract archive in target_dir
    extracted_path = path.join(target_dir, extracted_data)
    if not path.exists(extracted_path):
        print('No directory "%s" - extracting archive...' % extracted_path)
        with tarfile.open(archive_path) as tar:
            tar.extractall(target_dir)
    else:
        print('Found directory "%s" - not extracting it from archive.' % extracted_path)

def _maybe_convert_sets(target_dir, extracted_data):
    extracted_dir = path.join(target_dir, extracted_data)
    for source_csv in glob(path.join(extracted_dir, '*.csv')):
        _maybe_convert_set(extracted_dir, source_csv, path.join(target_dir, os.path.split(source_csv)[-1]))

def _maybe_convert_set(extracted_dir, source_csv, target_csv):
    print()
    if path.exists(target_csv):
        print('Found CSV file "%s" - not importing "%s".' % (target_csv, source_csv))
        return
    print('No CSV file "%s" - importing "%s"...' % (target_csv, source_csv))
    samples = []
    with open(source_csv) as source_csv_file:
        reader = csv.DictReader(source_csv_file)
        for row in reader:
            samples.append((row['filename'], row['text']))

    # Mutable counters for the concurrent embedded routine
    counter = { 'all': 0, 'failed': 0, 'invalid_label': 0, 'too_short': 0, 'too_long': 0 }
    lock = RLock()
    num_samples = len(samples)
    rows = []

    def one_sample(sample):
        mp3_filename = path.join(*(sample[0].split('/')))
        mp3_filename = path.join(extracted_dir, mp3_filename)
        # Storing wav files next to the mp3 ones - just with a different suffix
        wav_filename = path.splitext(mp3_filename)[0] + ".wav"
        _maybe_convert_wav(mp3_filename, wav_filename)
        frames = int(subprocess.check_output(['soxi', '-s', wav_filename], stderr=subprocess.STDOUT))
        file_size = -1
        if path.exists(wav_filename):
            file_size = path.getsize(wav_filename)
            frames = int(subprocess.check_output(['soxi', '-s', wav_filename], stderr=subprocess.STDOUT))
        label = validate_label(sample[1])
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

    print('Importing mp3 files...')
    pool = Pool(cpu_count())
    bar = progressbar.ProgressBar(max_value=num_samples, widgets=SIMPLE_BAR)
    for i, _ in enumerate(pool.imap_unordered(one_sample, samples), start=1):
        bar.update(i)
    bar.update(num_samples)
    pool.close()
    pool.join()

    print('Writing "%s"...' % target_csv)
    with open(target_csv, 'w') as target_csv_file:
        writer = csv.DictWriter(target_csv_file, fieldnames=FIELDNAMES)
        writer.writeheader()
        bar = progressbar.ProgressBar(max_value=len(rows), widgets=SIMPLE_BAR)
        for filename, file_size, transcript in bar(rows):
            writer.writerow({ 'wav_filename': filename, 'wav_filesize': file_size, 'transcript': transcript })

    print('Imported %d samples.' % (counter['all'] - counter['failed'] - counter['too_short'] - counter['too_long']))
    if counter['failed'] > 0:
        print('Skipped %d samples that failed upon conversion.' % counter['failed'])
    if counter['invalid_label'] > 0:
        print('Skipped %d samples that failed on transcript validation.' % counter['invalid_label'])
    if counter['too_short'] > 0:
        print('Skipped %d samples that were too short to match the transcript.' % counter['too_short'])
    if counter['too_long'] > 0:
        print('Skipped %d samples that were longer than %d seconds.' % (counter['too_long'], MAX_SECS))

def _maybe_convert_wav(mp3_filename, wav_filename):
    if not path.exists(wav_filename):
        transformer = sox.Transformer()
        transformer.convert(samplerate=SAMPLE_RATE)
        try:
            transformer.build(mp3_filename, wav_filename)
        except sox.core.SoxError:
            pass

if __name__ == "__main__":
    _download_and_preprocess_data(sys.argv[1])
