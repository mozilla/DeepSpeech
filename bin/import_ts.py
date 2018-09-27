#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function

# Make sure we can import stuff from util/
# This script needs to be run from the root of the DeepSpeech repository
import os
import re
import sys


sys.path.insert(1, os.path.join(sys.path[0], '..'))

import csv
import zipfile

from os import path

from util.downloader import maybe_download
from util.text import validate_label

FIELDNAMES = ['wav_filename', 'wav_filesize', 'transcript']
MAX_SECS = 10
ARCHIVE_NAME = '2018-10-03_fr_FR'
ARCHIVE_DIR_NAME = 'ts_' + ARCHIVE_NAME
ARCHIVE_URL = 'https://s3.eu-west-3.amazonaws.com/audiocorp/releases/' + ARCHIVE_NAME + '.zip'


def _download_and_preprocess_data(target_dir):
    # Making path absolute
    target_dir = path.abspath(target_dir)
    # Conditionally download data
    archive_path = maybe_download('ts_' + ARCHIVE_NAME + '.zip', target_dir, ARCHIVE_URL)
    # Conditionally extract archive data
    _maybe_extract(target_dir, ARCHIVE_DIR_NAME, archive_path)
    # Conditionally convert TrainingSpeech data to DeepSpeech CSVs and wav
    _maybe_convert_sets(target_dir, ARCHIVE_DIR_NAME)

def _maybe_extract(target_dir, extracted_data, archive_path):
    # If target_dir/extracted_data does not exist, extract archive in target_dir
    extracted_path = path.join(target_dir, extracted_data)
    if not path.exists(extracted_path):
        print('No directory "%s" - extracting archive...' % extracted_path)
        if not os.path.isdir(extracted_path):
            os.mkdir(extracted_path)
        with zipfile.ZipFile(archive_path) as zip_f:
            zip_f.extractall(extracted_path)
    else:
        print('Found directory "%s" - not extracting it from archive.' % archive_path)

def _maybe_convert_sets(target_dir, extracted_data):
    extracted_dir = path.join(target_dir, extracted_data)
    # override existing CSV with normalized one
    target_csv_template = os.path.join(target_dir, 'ts_' + ARCHIVE_NAME + '_{}.csv')
    if os.path.isfile(target_csv_template):
        return
    path_to_original_csv = os.path.join(extracted_dir, 'data.csv')
    with open(path_to_original_csv) as csv_f:
        data = [
            d for d in csv.DictReader(csv_f, delimiter=',')
            if float(d['duration']) <= MAX_SECS
        ]
    with open(target_csv_template.format('train'), 'w') as train_csv_file:  # 80%
        with open(target_csv_template.format('dev'), 'w') as dev_csv_file:  # 10%
            with open(target_csv_template.format('test'), 'w') as test_csv_file:  # 10%
                train_writer = csv.DictWriter(train_csv_file, fieldnames=FIELDNAMES)
                train_writer.writeheader()
                dev_writer = csv.DictWriter(dev_csv_file, fieldnames=FIELDNAMES)
                dev_writer.writeheader()
                test_writer = csv.DictWriter(test_csv_file, fieldnames=FIELDNAMES)
                test_writer.writeheader()

                for i, item in enumerate(data):
                    transcript = validate_label(cleanup_transcript(item['text']))
                    if not transcript:
                        continue
                    wav_filename = os.path.join(target_dir, extracted_data, item['path'])
                    i_mod = i % 10
                    if i_mod == 0:
                        writer = test_writer
                    elif i_mod == 1:
                        writer = dev_writer
                    else:
                        writer = train_writer
                    writer.writerow(dict(
                        wav_filename=wav_filename,
                        wav_filesize=os.path.getsize(wav_filename),
                        transcript=transcript,
                    ))


PUNCTUATIONS_REG = re.compile(r"[°\-,;!?.()\[\]*…—]")
MULTIPLE_SPACES_REG = re.compile(r'\s{2,}')


def cleanup_transcript(text):
    text = text.replace('’', "'").replace('\u00A0', ' ')
    text = PUNCTUATIONS_REG.sub(' ', text)
    text = MULTIPLE_SPACES_REG.sub(' ', text)
    return text.strip().lower()


if __name__ == "__main__":
    _download_and_preprocess_data(sys.argv[1])
