from __future__ import absolute_import, division, print_function

import tensorflow as tf
import pandas

from math import ceil
from os import path
from six.moves import range
from tensorflow.contrib.learn.python.learn.datasets import base
from threading import Thread
from util.audio import audiofile_to_input_vector
from util.data_set_helpers import DataSets
from util.gpu import get_available_gpus
from util.text import text_to_char_array, ctc_label_dense_to_sparse

class DataSet(object):
    def __init__(self, filelist, thread_count, batch_size, numcep, numcontext, next_index=lambda x: x + 1):
        self._coord = None
        self._numcep = numcep
        self._x = tf.placeholder(tf.float32, [None, numcep + (2 * numcep * numcontext)])
        self._x_length = tf.placeholder(tf.int32, [])
        self._y = tf.placeholder(tf.int32, [None,])
        self._y_length = tf.placeholder(tf.int32, [])
        self.example_queue = tf.PaddingFIFOQueue(shapes=[[None, numcep + (2 * numcep * numcontext)], [], [None,], []],
                                                  dtypes=[tf.float32, tf.int32, tf.int32, tf.int32],
                                                  capacity=2 * self._get_device_count() * batch_size)
        self._enqueue_op = self.example_queue.enqueue([self._x, self._x_length, self._y, self._y_length])
        self._close_op = self.example_queue.close(cancel_pending_enqueues=True)
        self._filelist = filelist
        self.batch_size = batch_size
        self._numcontext = numcontext
        self._thread_count = thread_count
        self._files_list = self._create_files_list()
        self._next_index = next_index

    def _get_device_count(self):
        available_gpus = get_available_gpus()
        return max(len(available_gpus), 1)

    def start_queue_threads(self, session, coord):
        self._coord = coord
        batch_threads = [Thread(target=self._populate_batch_queue, args=(session,)) for i in range(self._thread_count)]
        for batch_thread in batch_threads:
            batch_thread.daemon = True
            batch_thread.start()
        return batch_threads

    def close_queue(self, session):
        session.run(self._close_op)

    def _create_files_list(self):
        # 1. Sort by wav filesize
        # 2. Select just wav filename and transcript columns
        # 3. Return a NumPy representation
        return self._filelist.sort_values(by="wav_filesize")        \
                             .ix[:, ["wav_filename", "transcript"]] \
                             .values

    def _indices(self):
        index = -1
        while not self._coord.should_stop():
            index = self._next_index(index) % len(self._files_list)
            yield self._files_list[index]

    def _populate_batch_queue(self, session):
        for wav_file, transcript in self._indices():
            source = audiofile_to_input_vector(wav_file, self._numcep, self._numcontext)
            source_len = len(source)
            target = text_to_char_array(transcript)
            target_len = len(target)
            try:
                session.run(self._enqueue_op, feed_dict={
                    self._x: source,
                    self._x_length: source_len,
                    self._y: target,
                    self._y_length: target_len})
            except tf.errors.CancelledError:
                return

    def next_batch(self):
        source, source_lengths, target, target_lengths = self.example_queue.dequeue_many(self.batch_size)
        sparse_labels = ctc_label_dense_to_sparse(target, target_lengths, self.batch_size)
        return source, source_lengths, sparse_labels

    @property
    def total_batches(self):
        # Note: If len(_filelist) % batch_size != 0, this re-uses initial files
        return int(ceil(len(self._filelist) / self.batch_size))


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
