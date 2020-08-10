#!/usr/bin/env python3
import argparse
import functools
import pandas

from mozilla_voice_stt_training.util.helpers import secs_to_hours
from pathlib import Path


def read_csvs(csv_files):
    # Relative paths are relative to CSV location
    def absolutify(csv, path):
        path = Path(path)
        if path.is_absolute():
            return str(path)
        return str(csv.parent / path)

    sets = []
    for csv in csv_files:
        file = pandas.read_csv(csv, encoding='utf-8', na_filter=False)
        file['wav_filename'] = file['wav_filename'].apply(functools.partial(absolutify, csv))
        sets.append(file)

    # Concat all sets, drop any extra columns, re-index the final result as 0..N
    return pandas.concat(sets, join='inner', ignore_index=True)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-csv", "--csv-files", help="Str. Filenames as a comma separated list", required=True)
    parser.add_argument("--sample-rate", type=int, default=16000, required=False, help="Audio sample rate")
    parser.add_argument("--channels", type=int, default=1, required=False, help="Audio channels")
    parser.add_argument("--bits-per-sample", type=int, default=16, required=False, help="Audio bits per sample")
    args = parser.parse_args()
    in_files = [Path(i).absolute() for i in args.csv_files.split(",")]

    csv_dataframe = read_csvs(in_files)
    total_bytes = csv_dataframe['wav_filesize'].sum()
    total_files = len(csv_dataframe)
    total_seconds = ((csv_dataframe['wav_filesize'] - 44) / args.sample_rate / args.channels / (args.bits_per_sample // 8)).sum()

    print('Total bytes:', total_bytes)
    print('Total files:', total_files)
    print('Total time:', secs_to_hours(total_seconds))

if __name__ == '__main__':
    main()
