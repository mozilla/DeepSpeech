#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

# Make sure we can import stuff from util/
# This script needs to be run from the root of the DeepSpeech repository
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import pandas

from util.downloader import maybe_download

def _download_and_preprocess_data(data_dir):
    # Conditionally download data
    JRM_BASE="http://jrmeyer.github.io/misc/ru-deepspeech/"
    local_file = maybe_download("ru.wav", data_dir, JRM_BASE + "ru.wav")
    csv_file = maybe_download("ru.csv", data_dir, JRM_BASE + "ru.csv")
    alphabet_file = maybe_download("alphabet.ru", data_dir, JRM_BASE + "alphabet.ru")

if __name__ == "__main__":
    _download_and_preprocess_data(sys.argv[1])
