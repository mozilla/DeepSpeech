from __future__ import absolute_import, division, print_function

import tensorflow as tf
import pandas

from os import path
from tensorflow.contrib.learn.python.learn.datasets import base
from util.data_set_helpers import DataSets, DataSet

def read_data_sets(data_dir, train_csvs, dev_csvs, test_csvs,
                   train_batch_size, dev_batch_size, test_batch_size,
                   numcep, numcontext, thread_count=8,
                   stride=1, offset=0, next_index=lambda s, i: i + 1,
                   limit_dev=0, limit_test=0, limit_train=0, sets=[]):
    # Read the processed set files from disk if they exist, otherwise create them.
    def read_csvs(csvs):
        files = None
        for csv in csvs:
            file = pandas.read_csv(csv)
            if files is None:
                files = file
            else:
                files = files.append(file)
        return files

    train_files = read_csvs(train_csvs)
    dev_files = read_csvs(dev_csvs)
    test_files = read_csvs(test_csvs)

    if train_files is None or dev_files is None or test_files is None:
        # Conditionally download data
        LDC93S1_BASE = "LDC93S1"
        LDC93S1_BASE_URL = "https://catalog.ldc.upenn.edu/desc/addenda/"
        local_file = base.maybe_download(LDC93S1_BASE + ".wav", data_dir, LDC93S1_BASE_URL + LDC93S1_BASE + ".wav")
        trans_file = base.maybe_download(LDC93S1_BASE + ".txt", data_dir, LDC93S1_BASE_URL + LDC93S1_BASE + ".txt")
        with open(trans_file, "r") as fin:
            transcript = ' '.join(fin.read().strip().lower().split(' ')[2:]).replace('.', '')

        df = pandas.DataFrame(data=[(local_file, path.getsize(local_file), transcript)],
                              columns=["wav_filename", "wav_filesize", "transcript"])
        df.to_csv(path.join(data_dir, "ldc93s1.csv"), index=False)

        train_files = dev_files = test_files = df

    # Create train DataSet
    train = None
    if "train" in sets:
        train = DataSet(train_files, thread_count, train_batch_size, numcep, numcontext)

    # Create dev DataSet
    dev = None
    if "dev" in sets:
        dev = DataSet(dev_files, thread_count, dev_batch_size, numcep, numcontext)

    # Create test DataSet
    test = None
    if "test" in sets:
        test = DataSet(test_files, thread_count, test_batch_size, numcep, numcontext)

    # Return DataSets
    return DataSets(train, dev, test)
