#!/usr/bin/env python
import os
import sys

import pandas

from deepspeech_training.util.downloader import maybe_download


def _download_and_preprocess_data(data_dir):
    # Conditionally download data
    LDC93S1_BASE = "LDC93S1"
    LDC93S1_BASE_URL = "https://catalog.ldc.upenn.edu/desc/addenda/"
    local_file = maybe_download(
        LDC93S1_BASE + ".wav", data_dir, LDC93S1_BASE_URL + LDC93S1_BASE + ".wav"
    )
    trans_file = maybe_download(
        LDC93S1_BASE + ".txt", data_dir, LDC93S1_BASE_URL + LDC93S1_BASE + ".txt"
    )
    with open(trans_file, "r") as fin:
        transcript = " ".join(fin.read().strip().lower().split(" ")[2:]).replace(
            ".", ""
        )

    df = pandas.DataFrame(
        data=[(os.path.abspath(local_file), os.path.getsize(local_file), transcript)],
        columns=["wav_filename", "wav_filesize", "transcript"],
    )
    df.to_csv(os.path.join(data_dir, "ldc93s1.csv"), index=False)


if __name__ == "__main__":
    _download_and_preprocess_data(sys.argv[1])
