#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

# Make sure we can import stuff from util/
# This script needs to be run from the root of the DeepSpeech repository
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import csv
import tarfile
import subprocess
from glob import glob
from os import path
from sox import Transformer
from threading import Lock
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.platform import gfile
from multiprocessing.dummy import Pool
from multiprocessing import cpu_count
from util.progress import print_progress

FIELDNAMES = ["wav_filename", "wav_filesize", "transcript"]
SAMPLE_RATE = 16000
MAX_SECS = 10
CV_DIR = "cv_corpus_v1"
CV_DATA = "cv_corpus_v1.tar.gz"
CV_DATA_URL = "https://s3.us-east-2.amazonaws.com/common-voice-data-download/cv_corpus_v1.tar.gz"

def _download_and_preprocess_data(data_dir):
    # Making path absolute
    data_dir = path.abspath(data_dir)
    # Conditionally download data
    local_file = base.maybe_download(CV_DATA, data_dir, CV_DATA_URL)
    # Conditionally extract common voice data
    _maybe_extract(data_dir, CV_DIR, local_file)
    # Conditionally convert common voice CSV files and mp3 data to DeepSpeech CSVs and wav
    _maybe_convert_sets(data_dir, CV_DIR)

def _maybe_extract(data_dir, extracted_data, archive):
    # If data_dir/extracted_data does not exist, extract archive in data_dir
    if not gfile.Exists(path.join(data_dir, extracted_data)):
        print('Extracting "%s"...' % archive)
        with tarfile.open(archive) as tar:
            members = list(tar.getmembers())
            for i, member in enumerate(members):
                print_progress(i + 1, len(members))
                tar.extract(member, path=data_dir)

def _maybe_convert_sets(data_dir, extracted_data):
    extracted_dir = path.join(data_dir, extracted_data)
    for source_csv in glob(path.join(extracted_dir, '*.csv')):
        _maybe_convert_set(extracted_dir, source_csv, path.join(data_dir, os.path.split(source_csv)[-1]))
        
def _maybe_convert_set(extracted_dir, source_csv, target_csv):
    if gfile.Exists(target_csv):
        return
    print('Importing "%s" and its listed mp3 files...' % source_csv)
    samples = []
    with open(source_csv) as source_csv_file:
        reader = csv.DictReader(source_csv_file)
        for row in reader:
            samples.append((row['filename'], row['text']))

    with open(target_csv, 'w') as target_csv_file:
        writer = csv.DictWriter(target_csv_file, fieldnames=FIELDNAMES)
        writer.writeheader()
        # Mutable counters for the concurrent embedded routine
        counter = { 'all': 0, 'too_short': 0, 'too_long': 0 }
        lock = Lock()
        num_samples = len(samples)

        def one_sample(sample):
            mp3_filename = path.join(*(sample[0].split('/')))
            mp3_filename = path.join(extracted_dir, mp3_filename)
            # Storing wav files next to the mp3 ones - just with a different suffix
            wav_filename = path.splitext(mp3_filename)[0] + ".wav"
            _maybe_convert_wav(mp3_filename, wav_filename)
            frames = int(subprocess.check_output(['soxi', '-s', wav_filename], stderr=subprocess.STDOUT))
            if int(frames/SAMPLE_RATE*1000/10/2) < len(str(sample[1])):
                # Excluding samples that are too short to fit the transcript
                counter['too_short'] += 1
            elif frames/SAMPLE_RATE > MAX_SECS:
                # Excluding very long samples to keep a reasonable batch-size
                counter['too_long'] += 1
            else:
                writer.writerow({ 'wav_filename': wav_filename,
                                  'wav_filesize': path.getsize(wav_filename),
                                  'transcript': sample[1] })
            counter['all'] += 1
            with lock: 
                print_progress(counter['all'], num_samples)

        pool = Pool(cpu_count())
        pool.map(one_sample, samples)
        pool.close()
        pool.join()
    
        print('Imported %d samples.' % (counter['all'] - counter['too_short'] - counter['too_long']))
        if counter['too_short'] > 0:
            print('Skipped %d samples that were too short to match the transcript.' % counter['too_short'])
        if counter['too_long'] > 0:
            print('Skipped %d samples that were longer than %d seconds.' % (counter['too_long'], MAX_SECS))

def _maybe_convert_wav(mp3_filename, wav_filename):
    if not gfile.Exists(wav_filename):
        transformer = Transformer()
        transformer.convert(samplerate=SAMPLE_RATE)
        transformer.build(mp3_filename, wav_filename)

if __name__ == "__main__":
    _download_and_preprocess_data(sys.argv[1])
