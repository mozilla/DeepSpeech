#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function

# Make sure we can import stuff from util/
# This script needs to be run from the root of the DeepSpeech repository
import os
import re
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from util.importers import get_importers_parser, get_validate_label

import csv
import unidecode
import zipfile
import sox
import subprocess
import progressbar

from threading import RLock
from multiprocessing.dummy import Pool
from multiprocessing import cpu_count
from util.downloader import SIMPLE_BAR

from os import path

from util.downloader import maybe_download
from util.helpers import secs_to_hours

FIELDNAMES = ['wav_filename', 'wav_filesize', 'transcript']
SAMPLE_RATE = 16000
MAX_SECS = 15
ARCHIVE_NAME = '2019-04-11_fr_FR'
ARCHIVE_DIR_NAME = 'ts_' + ARCHIVE_NAME
ARCHIVE_URL = 'https://deepspeech-storage-mirror.s3.fr-par.scw.cloud/' + ARCHIVE_NAME + '.zip'


def _download_and_preprocess_data(target_dir, english_compatible=False):
    # Making path absolute
    target_dir = path.abspath(target_dir)
    # Conditionally download data
    archive_path = maybe_download('ts_' + ARCHIVE_NAME + '.zip', target_dir, ARCHIVE_URL)
    # Conditionally extract archive data
    _maybe_extract(target_dir, ARCHIVE_DIR_NAME, archive_path)
    # Conditionally convert TrainingSpeech data to DeepSpeech CSVs and wav
    _maybe_convert_sets(target_dir, ARCHIVE_DIR_NAME, english_compatible=english_compatible)


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


def _maybe_convert_sets(target_dir, extracted_data, english_compatible=False):
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

    # Keep track of how many samples are good vs. problematic
    counter = {'all': 0, 'failed': 0, 'invalid_label': 0, 'too_short': 0, 'too_long': 0, 'total_time': 0}
    lock = RLock()
    num_samples = len(data)
    rows = []

    wav_root_dir = extracted_dir

    def one_sample(sample):
        """ Take a audio file, and optionally convert it to 16kHz WAV """
        orig_filename = path.join(wav_root_dir, sample['path'])
        # Storing wav files next to the wav ones - just with a different suffix
        wav_filename = path.splitext(orig_filename)[0] + ".converted.wav"
        _maybe_convert_wav(orig_filename, wav_filename)
        file_size = -1
        frames = 0
        if path.exists(wav_filename):
            file_size = path.getsize(wav_filename)
            frames = int(subprocess.check_output(['soxi', '-s', wav_filename], stderr=subprocess.STDOUT))
        label = sample['text']
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

    print("Importing wav files...")
    pool = Pool(cpu_count())
    bar = progressbar.ProgressBar(max_value=num_samples, widgets=SIMPLE_BAR)
    for i, _ in enumerate(pool.imap_unordered(one_sample, data), start=1):
        bar.update(i)
    bar.update(num_samples)
    pool.close()
    pool.join()

    with open(target_csv_template.format('train'), 'w') as train_csv_file:  # 80%
        with open(target_csv_template.format('dev'), 'w') as dev_csv_file:  # 10%
            with open(target_csv_template.format('test'), 'w') as test_csv_file:  # 10%
                train_writer = csv.DictWriter(train_csv_file, fieldnames=FIELDNAMES)
                train_writer.writeheader()
                dev_writer = csv.DictWriter(dev_csv_file, fieldnames=FIELDNAMES)
                dev_writer.writeheader()
                test_writer = csv.DictWriter(test_csv_file, fieldnames=FIELDNAMES)
                test_writer.writeheader()

                for i, item in enumerate(rows):
                    print('item', item)
                    transcript = validate_label(cleanup_transcript(item[2], english_compatible=english_compatible))
                    if not transcript:
                        continue
                    wav_filename = os.path.join(target_dir, extracted_data, item[0])
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

def _maybe_convert_wav(orig_filename, wav_filename):
    if not path.exists(wav_filename):
        transformer = sox.Transformer()
        transformer.convert(samplerate=SAMPLE_RATE)
        try:
            transformer.build(orig_filename, wav_filename)
        except sox.core.SoxError as ex:
            print('SoX processing error', ex, orig_filename, wav_filename)


PUNCTUATIONS_REG = re.compile(r"[°\-,;!?.()\[\]*…—]")
MULTIPLE_SPACES_REG = re.compile(r'\s{2,}')


def cleanup_transcript(text, english_compatible=False):
    text = text.replace('’', "'").replace('\u00A0', ' ')
    text = PUNCTUATIONS_REG.sub(' ', text)
    text = MULTIPLE_SPACES_REG.sub(' ', text)
    if english_compatible:
        text = unidecode.unidecode(text)
    return text.strip().lower()


def handle_args():
    parser = get_importers_parser(description='Importer for TrainingSpeech dataset.')
    parser.add_argument(dest='target_dir')
    parser.add_argument('--english-compatible', action='store_true', dest='english_compatible', help='Remove diactrics and other non-ascii chars.')
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = handle_args()
    validate_label = get_validate_label(cli_args)
    _download_and_preprocess_data(cli_args.target_dir, cli_args.english_compatible)
