#!/usr/bin/env python

# VCTK used in wavenet paper https://arxiv.org/pdf/1609.03499.pdf

from __future__ import absolute_import, division, print_function

# Make sure we can import stuff from util/
# This script needs to be run from the root of the DeepSpeech repository
import os
import random
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import re
import csv
import librosa
import subprocess
import progressbar

from glob import glob
from os import path
from threading import RLock
from multiprocessing.dummy import Pool
from multiprocessing import cpu_count
from util.text import validate_label
from util.downloader import maybe_download, SIMPLE_BAR
from zipfile import ZipFile

SAMPLE_RATE = 16000
MAX_SECS = 10
MIN_SECS = 1
ARCHIVE_DIR_NAME = 'VCTK-Corpus'
ARCHIVE_NAME = 'VCTK-Corpus.zip?sequence=2&isAllowed=y'
ARCHIVE_URL = 'https://datashare.is.ed.ac.uk/bitstream/handle/10283/2651/' + ARCHIVE_NAME

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
        print(f'No directory {extracted_path} - extracting archive...')
        with ZipFile(archive_path, 'r') as zipObj:
            # Extract all the contents of zip file in current directory
            zipObj.extractall(target_dir)
    else:
        print(f'Found directory {extracted_path} - not extracting it from archive.')

def _maybe_convert_sets(target_dir, extracted_data):
    extracted_dir = path.join(target_dir, extracted_data, 'wav48')
    txt_dir = path.join(target_dir, extracted_data, 'txt')

    cnt = 1
    dir = os.path.expanduser(extracted_dir)
    srtd = len(sorted(os.listdir(dir)))
              
    for target in sorted(os.listdir(dir)):
        print(f'\nSpeaker {cnt} of {srtd}')
        _maybe_convert_set(extracted_dir, txt_dir, target_dir, path.join(extracted_dir, os.path.split(target)[-1]))
        cnt += 1
    
    _write_csv(extracted_dir, txt_dir, target_dir)
    
def _maybe_convert_set(extracted_dir, txt_dir, target_dir, target_csv):
    
    def one_sample(sample):
        if is_audio_file(sample):
            sample = os.path.join(target_csv,sample)

            y, sr = librosa.load(sample, sr=16000)

            # Trim the beginning and ending silence
            yt, index = librosa.effects.trim(y)

            duration = librosa.get_duration(yt,sr)
            if duration > MAX_SECS or duration < MIN_SECS:
                os.remove(sample)
            else:
                librosa.output.write_wav(sample, yt, sr)
    
    samples = sorted(os.listdir(target_csv))
    
    lock = RLock()
    num_samples = len(samples)
    rows = []   

    print(f'Converting wav files to {SAMPLE_RATE}hz...')
    pool = Pool(cpu_count())
    bar = progressbar.ProgressBar(max_value=num_samples, widgets=SIMPLE_BAR)
    for i, _ in enumerate(pool.imap_unordered(one_sample, samples), start=1):
        bar.update(i)
    bar.update(num_samples)
    pool.close()
    pool.join()
        

def _write_csv(extracted_dir, txt_dir, target_dir):
    print(f'Writing CSV file')
    print(target_dir)
    dset_abs_path = extracted_dir
    dset_txt_abs_path = txt_dir

    audios = make_manifest(dset_abs_path)
    utterences = load_txts(dset_txt_abs_path)

    csv = []

    for file in audios:

        st = os.stat(file)
        file_size = st.st_size
        
        #Seems to be one wav directory missing from txts - skip it
        file_parts = file.split(os.sep)
        file_subdir = file_parts[-2]
        if file_subdir == 'p315': 
            continue

        file_name = file_parts[-1]
        file_name_no_ext = file_name.split('.')[0]

        utterence = utterences[file_name_no_ext]
        utterence_clean = re.sub(r"[^a-zA-Z' ]+", "", utterence).lower().strip()

        csv_line = f'{file},{file_size},{utterence_clean}\n'
        csv.append(csv_line)

    
    random.shuffle(csv)

    train_data = csv[:37000]
    dev_data = csv[37000:40200]
    test_data = csv[40200:]

    print(len(train_data))
    print(len(dev_data))
    print(len(test_data))

    with open(os.path.join(extracted_dir,'vctk_full.csv'),'w') as fd:
        fd.write('wav_filename,wav_filesize,transcript\n')
        for i in csv:
            fd.write(i)
    with open(os.path.join(extracted_dir,'vctk_train.csv'),'w') as fd:
        fd.write('wav_filename,wav_filesize,transcript\n')
        for i in train_data:
            fd.write(i)
    with open(os.path.join(extracted_dir,'vctk_dev.csv'),'w') as fd:
        fd.write('wav_filename,wav_filesize,transcript\n')
        for i in dev_data:
            fd.write(i)
    with open(os.path.join(extracted_dir,'vctk_test.csv'),'w') as fd:
        fd.write('wav_filename,wav_filesize,transcript\n')
        for i in test_data:
            fd.write(i)
            
    print(f'Wrote {len(csv)} entries')

def make_manifest(dir):
    audios = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in fnames:
                path = os.path.join(root, fname)
                item = path
                audios.append(item)
    return audios


def load_txts(dir):
    utterences = dict()
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in fnames:
                if fname.endswith(".txt"):
                    with open(os.path.join(root, fname), "r") as f:
                        fname_no_ext = os.path.basename(
                            fname).rsplit(".", 1)[0]
                        utterences[fname_no_ext] = f.readline()
    return utterences

AUDIO_EXTENSIONS = [
    '.wav', 'WAV'
]


def is_audio_file(filename):
    return any(filename.endswith(extension) for extension in AUDIO_EXTENSIONS)


if __name__ == "__main__":
    _download_and_preprocess_data(sys.argv[1])
