#!/usr/bin/env python3

import argparse
import os

from util.helpers import secs_to_hours
from util.feeding import read_csvs

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-csv", "--csv-files", help="Str. Filenames as a comma separated list", required=True)
    parser.add_argument("--sample-rate", type=int, default=16000, required=False, help="Audio sample rate")
    parser.add_argument("--channels", type=int, default=1, required=False, help="Audio channels")
    parser.add_argument("--bits-per-sample", type=int, default=16, required=False, help="Audio bits per sample")
    args = parser.parse_args()
    in_files = [os.path.abspath(i) for i in args.csv_files.split(",")]

    csv_dataframe = read_csvs(in_files)
    total_bytes = csv_dataframe['wav_filesize'].sum()
    total_files = len(csv_dataframe.index)

    bytes_without_headers = total_bytes - 44 * total_files

    total_time = bytes_without_headers / (args.sample_rate * args.channels * args.bits_per_sample / 8)

    print('total_bytes', total_bytes)
    print('total_files', total_files)
    print('bytes_without_headers', bytes_without_headers)
    print('total_time', secs_to_hours(total_time))

if __name__ == '__main__':
    main()
