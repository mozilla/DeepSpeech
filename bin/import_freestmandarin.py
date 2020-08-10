#!/usr/bin/env python
import glob
import os
import tarfile

import numpy as np
import pandas

from mozilla_voice_stt_training.util.importers import get_importers_parser

COLUMN_NAMES = ["wav_filename", "wav_filesize", "transcript"]


def extract(archive_path, target_dir):
    print("Extracting {} into {}...".format(archive_path, target_dir))
    with tarfile.open(archive_path) as tar:
        tar.extractall(target_dir)


def preprocess_data(tgz_file, target_dir):
    # First extract main archive and sub-archives
    extract(tgz_file, target_dir)
    main_folder = os.path.join(target_dir, "ST-CMDS-20170001_1-OS")

    # Folder structure is now:
    # - ST-CMDS-20170001_1-OS/
    #   - *.wav
    #   - *.txt
    #   - *.metadata

    def load_set(glob_path):
        set_files = []
        for wav in glob.glob(glob_path):
            wav_filename = wav
            wav_filesize = os.path.getsize(wav)
            txt_filename = os.path.splitext(wav_filename)[0] + ".txt"
            with open(txt_filename, "r") as fin:
                transcript = fin.read()
            set_files.append((wav_filename, wav_filesize, transcript))
        return set_files

    # Load all files, then deterministically split into train/dev/test sets
    all_files = load_set(os.path.join(main_folder, "*.wav"))
    df = pandas.DataFrame(data=all_files, columns=COLUMN_NAMES)
    df.sort_values(by="wav_filename", inplace=True)

    indices = np.arange(0, len(df))
    np.random.seed(12345)
    np.random.shuffle(indices)

    # Total corpus size: 102600 samples. 5000 samples gives us 99% confidence
    # level with a margin of error of under 2%.
    test_indices = indices[-5000:]
    dev_indices = indices[-10000:-5000]
    train_indices = indices[:-10000]

    train_files = df.iloc[train_indices]
    durations = (train_files["wav_filesize"] - 44) / 16000 / 2
    train_files = train_files[durations <= 10.0]
    print("Trimming {} samples > 10 seconds".format((durations > 10.0).sum()))
    dest_csv = os.path.join(target_dir, "freestmandarin_train.csv")
    print("Saving train set into {}...".format(dest_csv))
    train_files.to_csv(dest_csv, index=False)

    dev_files = df.iloc[dev_indices]
    dest_csv = os.path.join(target_dir, "freestmandarin_dev.csv")
    print("Saving dev set into {}...".format(dest_csv))
    dev_files.to_csv(dest_csv, index=False)

    test_files = df.iloc[test_indices]
    dest_csv = os.path.join(target_dir, "freestmandarin_test.csv")
    print("Saving test set into {}...".format(dest_csv))
    test_files.to_csv(dest_csv, index=False)


def main():
    # https://www.openslr.org/38/
    parser = get_importers_parser(description="Import Free ST Chinese Mandarin corpus")
    parser.add_argument("tgz_file", help="Path to ST-CMDS-20170001_1-OS.tar.gz")
    parser.add_argument(
        "--target_dir",
        default="",
        help="Target folder to extract files into and put the resulting CSVs. Defaults to same folder as the main archive.",
    )
    params = parser.parse_args()

    if not params.target_dir:
        params.target_dir = os.path.dirname(params.tgz_file)

    preprocess_data(params.tgz_file, params.target_dir)


if __name__ == "__main__":
    main()
