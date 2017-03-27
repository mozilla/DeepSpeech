from __future__ import absolute_import
import tensorflow as tf

from os import path
from glob import glob
from math import ceil
from threading import Thread
from util.gpu import get_available_gpus
from util.text import text_to_char_array, ctc_label_dense_to_sparse
from util.audio import audiofile_to_input_vector
from tensorflow.contrib.learn.python.learn.datasets import base
from six.moves import range

class DataSets(object):
    def __init__(self, train, dev, test):
        self._dev = dev
        self._test = test
        self._train = train

    def start_queue_threads(self, sesssion):
        self._dev.start_queue_threads(sesssion)
        self._test.start_queue_threads(sesssion)
        self._train.start_queue_threads(sesssion)

    @property
    def train(self):
        return self._train

    @property
    def dev(self):
        return self._dev

    @property
    def test(self):
        return self._test

class DataSet(object):
    def __init__(self, txt_files, thread_count, batch_size, numcep, numcontext):
        self._coord = None
        self._numcep = numcep
        self._x = tf.placeholder(tf.float32, [None, numcep + (2 * numcep * numcontext)])
        self._x_length = tf.placeholder(tf.int32, [])
        self._y = tf.placeholder(tf.int32, [None,])
        self._y_length = tf.placeholder(tf.int32, [])
        self._example_queue = tf.PaddingFIFOQueue(shapes=[[None, numcep + (2 * numcep * numcontext)], [], [None,], []],
                                                  dtypes=[tf.float32, tf.int32, tf.int32, tf.int32],
                                                  capacity=2 * self._get_device_count() * batch_size)
        self._enqueue_op = self._example_queue.enqueue([self._x, self._x_length, self._y, self._y_length])
        self._close_op = self._example_queue.close(cancel_pending_enqueues=True)
        self._txt_files = txt_files
        self._batch_size = batch_size
        self._numcontext = numcontext
        self._thread_count = thread_count

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

    def _compute_source_target(self):
        txt_file = self._txt_files[0]
        wav_file = path.splitext(txt_file)[0] + ".wav"

        audio_waves = audiofile_to_input_vector(wav_file, self._numcep, self._numcontext)

        with open(txt_file) as open_txt_file:
            original = ' '.join(open_txt_file.read().strip().lower().split(' ')[2:]).replace('.', '')

        target = text_to_char_array(original)

        return audio_waves, len(audio_waves), target, len(target)

    def _populate_batch_queue(self, session):
        source, source_len, target, target_len = self._compute_source_target()
        while not self._coord.should_stop():
            try:
                session.run(self._enqueue_op, feed_dict={
                    self._x: source,
                    self._x_length: source_len,
                    self._y: target,
                    self._y_length: target_len})
            except tf.errors.CancelledError:
                return

    def next_batch(self):
        source, source_lengths, target, target_lengths = self._example_queue.dequeue_many(self._batch_size)
        sparse_labels = ctc_label_dense_to_sparse(target, target_lengths, self._batch_size)
        return source, source_lengths, sparse_labels

    @property
    def total_batches(self):
        # Note: If len(_txt_files) % _batch_size != 0, this re-uses initial _txt_files
        return int(ceil(float(len(self._txt_files)) /float(self._batch_size)))


def read_data_sets(data_dir, train_batch_size, dev_batch_size, test_batch_size, numcep, numcontext, thread_count=1, limit_dev=0, limit_test=0, limit_train=0, sets=[]):
    # Conditionally download data
    LDC93S1_BASE = "LDC93S1"
    LDC93S1_BASE_URL = "https://catalog.ldc.upenn.edu/desc/addenda/"
    local_file = base.maybe_download(LDC93S1_BASE + ".wav", data_dir, LDC93S1_BASE_URL + LDC93S1_BASE + ".wav")
    _ = base.maybe_download(LDC93S1_BASE + ".txt", data_dir, LDC93S1_BASE_URL + LDC93S1_BASE + ".txt")

    # Create all DataSets, we do not really need separation
    train = None
    if "train" in sets:
        train = _read_data_set(data_dir, thread_count, train_batch_size, numcep, numcontext)
    
    dev = None
    if "dev" in sets:
        dev   = _read_data_set(data_dir, thread_count, dev_batch_size, numcep, numcontext)
    
    test = None
    if "test" in sets:
        test  = _read_data_set(data_dir, thread_count, test_batch_size, numcep, numcontext)

    # Return DataSets
    return DataSets(train, dev, test)

def _read_data_set(data_dir, thread_count, batch_size, numcep, numcontext):
    # Obtain list of txt files
    txt_files = glob(path.join(data_dir, "*.txt"))

    # Return DataSet
    return DataSet(txt_files, thread_count, batch_size, numcep, numcontext)
