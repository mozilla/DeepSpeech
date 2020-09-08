#!/usr/bin/env python
import csv
import os
import sys
import subprocess
import tarfile
from glob import glob
from multiprocessing import Pool

import progressbar
import sox

from deepspeech_training.util.downloader import SIMPLE_BAR, maybe_download
from deepspeech_training.util.importers import (
    get_counter,
    get_imported_samples,
    print_import_report,
)
from deepspeech_training.util.importers import validate_label_eng as validate_label

FIELDNAMES = ["wav_filename", "wav_filesize", "transcript"]
SAMPLE_RATE = 16000
MAX_SECS = 10
ARCHIVE_DIR_NAME = "cv_corpus_v1"
ARCHIVE_NAME = ARCHIVE_DIR_NAME + ".tar.gz"
ARCHIVE_URL = (
    "https://s3.us-east-2.amazonaws.com/common-voice-data-download/" + ARCHIVE_NAME
)


def _download_and_preprocess_data(target_dir):
    # Making path absolute
    target_dir = os.path.abspath(target_dir)
    # Conditionally download data
    archive_path = maybe_download(ARCHIVE_NAME, target_dir, ARCHIVE_URL)
    # Conditionally extract common voice data
    _maybe_extract(target_dir, ARCHIVE_DIR_NAME, archive_path)
    # Conditionally convert common voice CSV files and mp3 data to DeepSpeech CSVs and wav
    _maybe_convert_sets(target_dir, ARCHIVE_DIR_NAME)


def _maybe_extract(target_dir, extracted_data, archive_path):
    # If target_dir/extracted_data does not exist, extract archive in target_dir
    extracted_path = os.join(target_dir, extracted_data)
    if not os.path.exists(extracted_path):
        print('No directory "%s" - extracting archive...' % extracted_path)
        with tarfile.open(archive_path) as tar:
            tar.extractall(target_dir)
    else:
        print('Found directory "%s" - not extracting it from archive.' % extracted_path)


def _maybe_convert_sets(target_dir, extracted_data):
    extracted_dir = os.path.join(target_dir, extracted_data)
    for source_csv in glob(os.path.join(extracted_dir, "*.csv")):
        _maybe_convert_set(
            extracted_dir,
            source_csv,
            os.path.join(target_dir, os.path.split(source_csv)[-1]),
        )


def one_sample(sample):
    mp3_filename = sample[0]
    # Storing wav files next to the mp3 ones - just with a different suffix
    wav_filename = path.splitext(mp3_filename)[0] + ".wav"
    _maybe_convert_wav(mp3_filename, wav_filename)
    frames = int(
        subprocess.check_output(["soxi", "-s", wav_filename], stderr=subprocess.STDOUT)
    )
    file_size = -1
    if os.path.exists(wav_filename):
        file_size = path.getsize(wav_filename)
        frames = int(
            subprocess.check_output(
                ["soxi", "-s", wav_filename], stderr=subprocess.STDOUT
            )
        )
    label = validate_label(sample[1])
    rows = []
    counter = get_counter()
    if file_size == -1:
        # Excluding samples that failed upon conversion
        counter["failed"] += 1
    elif label is None:
        # Excluding samples that failed on label validation
        counter["invalid_label"] += 1
    elif int(frames / SAMPLE_RATE * 1000 / 10 / 2) < len(str(label)):
        # Excluding samples that are too short to fit the transcript
        counter["too_short"] += 1
    elif frames / SAMPLE_RATE > MAX_SECS:
        # Excluding very long samples to keep a reasonable batch-size
        counter["too_long"] += 1
    else:
        # This one is good - keep it for the target CSV
        rows.append((wav_filename, file_size, label))
        counter["imported_time"] += frames
    counter["all"] += 1
    counter["total_time"] += frames
    return (counter, rows)


def _maybe_convert_set(extracted_dir, source_csv, target_csv):
    print()
    if os.path.exists(target_csv):
        print('Found CSV file "%s" - not importing "%s".' % (target_csv, source_csv))
        return
    print('No CSV file "%s" - importing "%s"...' % (target_csv, source_csv))
    samples = []
    with open(source_csv) as source_csv_file:
        reader = csv.DictReader(source_csv_file)
        for row in reader:
            samples.append((os.path.join(extracted_dir, row["filename"]), row["text"]))

    # Mutable counters for the concurrent embedded routine
    counter = get_counter()
    num_samples = len(samples)
    rows = []

    print("Importing mp3 files...")
    pool = Pool()
    bar = progressbar.ProgressBar(max_value=num_samples, widgets=SIMPLE_BAR)
    for i, processed in enumerate(pool.imap_unordered(one_sample, samples), start=1):
        counter += processed[0]
        rows += processed[1]
        bar.update(i)
    bar.update(num_samples)
    pool.close()
    pool.join()

    print('Writing "%s"...' % target_csv)
    with open(target_csv, "w", encoding="utf-8", newline="") as target_csv_file:
        writer = csv.DictWriter(target_csv_file, fieldnames=FIELDNAMES)
        writer.writeheader()
        bar = progressbar.ProgressBar(max_value=len(rows), widgets=SIMPLE_BAR)
        for filename, file_size, transcript in bar(rows):
            writer.writerow(
                {
                    "wav_filename": filename,
                    "wav_filesize": file_size,
                    "transcript": transcript,
                }
            )

    imported_samples = get_imported_samples(counter)
    assert counter["all"] == num_samples
    assert len(rows) == imported_samples

    print_import_report(counter, SAMPLE_RATE, MAX_SECS)


def _maybe_convert_wav(mp3_filename, wav_filename):
    if not os.path.exists(wav_filename):
        transformer = sox.Transformer()
        transformer.convert(samplerate=SAMPLE_RATE)
        try:
            transformer.build(mp3_filename, wav_filename)
        except sox.core.SoxError:
            pass


if __name__ == "__main__":
    _download_and_preprocess_data(sys.argv[1])
