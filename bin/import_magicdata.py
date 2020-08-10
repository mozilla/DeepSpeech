#!/usr/bin/env python
import glob
import os
import tarfile
import wave

import pandas

from mozilla_voice_stt_training.util.importers import get_importers_parser

COLUMN_NAMES = ["wav_filename", "wav_filesize", "transcript"]


def extract(archive_path, target_dir):
    print("Extracting {} into {}...".format(archive_path, target_dir))
    with tarfile.open(archive_path) as tar:
        tar.extractall(target_dir)


def is_file_truncated(wav_filename, wav_filesize):
    with wave.open(wav_filename, mode="rb") as fin:
        assert fin.getframerate() == 16000
        assert fin.getsampwidth() == 2
        assert fin.getnchannels() == 1

        header_duration = fin.getnframes() / fin.getframerate()
        filesize_duration = (wav_filesize - 44) / 16000 / 2

    return header_duration != filesize_duration


def preprocess_data(folder_with_archives, target_dir):
    # First extract subset archives
    for subset in ("train", "dev", "test"):
        extract(
            os.path.join(
                folder_with_archives, "magicdata_{}_set.tar.gz".format(subset)
            ),
            target_dir,
        )

    # Folder structure is now:
    # - magicdata_{train,dev,test}.tar.gz
    # - magicdata/
    #   - train/*.wav
    #   - train/TRANS.txt
    #   - dev/*.wav
    #   - dev/TRANS.txt
    #   - test/*.wav
    #   - test/TRANS.txt

    # The TRANS files are CSVs with three columns, one containing the WAV file
    # name, one containing the speaker ID, and one containing the transcription

    def load_set(set_path):
        transcripts = pandas.read_csv(
            os.path.join(set_path, "TRANS.txt"), sep="\t", index_col=0
        )
        glob_path = os.path.join(set_path, "*", "*.wav")
        set_files = []
        for wav in glob.glob(glob_path):
            try:
                wav_filename = wav
                wav_filesize = os.path.getsize(wav)
                transcript_key = os.path.basename(wav)
                transcript = transcripts.loc[transcript_key, "Transcription"]

                # Some files in this dataset are truncated, the header duration
                # doesn't match the file size. This causes errors at training
                # time, so check here if things are fine before including a file
                if is_file_truncated(wav_filename, wav_filesize):
                    print(
                        "Warning: File {} is corrupted, header duration does "
                        "not match file size. Ignoring.".format(wav_filename)
                    )
                    continue

                set_files.append((wav_filename, wav_filesize, transcript))
            except KeyError:
                print("Warning: Missing transcript for WAV file {}.".format(wav))
        return set_files

    for subset in ("train", "dev", "test"):
        print("Loading {} set samples...".format(subset))
        subset_files = load_set(os.path.join(target_dir, subset))
        df = pandas.DataFrame(data=subset_files, columns=COLUMN_NAMES)

        # Trim train set to under 10s
        if subset == "train":
            durations = (df["wav_filesize"] - 44) / 16000 / 2
            df = df[durations <= 10.0]
            print("Trimming {} samples > 10 seconds".format((durations > 10.0).sum()))

            with_noise = df["transcript"].str.contains(r"\[(FIL|SPK)\]")
            df = df[~with_noise]
            print(
                "Trimming {} samples with noise ([FIL] or [SPK])".format(
                    sum(with_noise)
                )
            )

        dest_csv = os.path.join(target_dir, "magicdata_{}.csv".format(subset))
        print("Saving {} set into {}...".format(subset, dest_csv))
        df.to_csv(dest_csv, index=False)


def main():
    # https://openslr.org/68/
    parser = get_importers_parser(description="Import MAGICDATA corpus")
    parser.add_argument(
        "folder_with_archives",
        help="Path to folder containing magicdata_{train,dev,test}.tar.gz",
    )
    parser.add_argument(
        "--target_dir",
        default="",
        help="Target folder to extract files into and put the resulting CSVs. Defaults to a folder called magicdata next to the archives",
    )
    params = parser.parse_args()

    if not params.target_dir:
        params.target_dir = os.path.join(params.folder_with_archives, "magicdata")

    preprocess_data(params.folder_with_archives, params.target_dir)


if __name__ == "__main__":
    main()
