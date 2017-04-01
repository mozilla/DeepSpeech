#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

# Make sure we can import stuff from util/
# This script needs to be run from the root of the DeepSpeech repository
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..', '..'))

import pandas

from tensorflow.contrib.learn.python.learn.datasets import base
from util.data_set_helpers import DataSets, DataSet

def _download_and_preprocess_data(data_dir):
    # Conditionally download data
    LDC93S1_BASE = "LDC93S1"
    LDC93S1_BASE_URL = "https://catalog.ldc.upenn.edu/desc/addenda/"
    local_file = base.maybe_download(LDC93S1_BASE + ".wav", data_dir, LDC93S1_BASE_URL + LDC93S1_BASE + ".wav")
    trans_file = base.maybe_download(LDC93S1_BASE + ".txt", data_dir, LDC93S1_BASE_URL + LDC93S1_BASE + ".txt")
    with open(trans_file, "r") as fin:
        transcript = ' '.join(fin.read().strip().lower().split(' ')[2:]).replace('.', '')

    df = pandas.DataFrame(data=[(local_file, os.path.getsize(local_file), transcript)],
                          columns=["wav_filename", "wav_filesize", "transcript"])
    df.to_csv(os.path.join(data_dir, "ldc93s1.csv"), index=False)

if __name__ == "__main__":
    _download_and_preprocess_data(sys.argv[1])
