import threading
import numpy as np

from os import path
from glob import glob
from math import ceil
from Queue import Queue
from threading import Thread
from util.gpu import get_available_gpus
from util.text import texts_to_sparse_tensor
from util.audio import audiofile_to_input_vector
from tensorflow.contrib.learn.python.learn.datasets import base

class DataSets(object):
    def __init__(self, train, dev, test):
        self._dev = dev
        self._test = test
        self._train = train

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
    def __init__(self, graph, txt_files, thread_count, batch_size, numcep, numcontext):
        self._graph = graph
        self._numcep = numcep
        self._batch_queue = Queue(2 * self._get_device_count())
        self._txt_files = txt_files
        self._batch_size = batch_size
        self._numcontext = numcontext
        self._thread_count = thread_count
        self._start_queue_threads()

    def _get_device_count(self):
        available_gpus = get_available_gpus()
        return max(len(available_gpus), 1)

    def _start_queue_threads(self):
        batch_threads = [Thread(target=self._populate_batch_queue) for i in xrange(self._thread_count)]
        for batch_thread in batch_threads:
            batch_thread.daemon = True
            batch_thread.start()

    def _compute_source_target(self):
        txt_file = self._txt_files[0]
        wav_file = path.splitext(txt_file)[0] + ".wav"

        audio_waves = audiofile_to_input_vector(wav_file, self._numcep, self._numcontext)
        
        with open(txt_file) as open_txt_file:
            original = ' '.join(open_txt_file.read().strip().lower().split(' ')[2:]).replace('.', '')

        source = np.array([ audio_waves for x in xrange(self._batch_size) ])
        target = texts_to_sparse_tensor([ original for x in xrange(self._batch_size) ])

        return source, target

    def _populate_batch_queue(self):
        with self._graph.as_default():
            source, target = self._compute_source_target()
            while True:
                self._batch_queue.put((source, target))

    def next_batch(self):
        source, target = self._batch_queue.get()
        return (source, target)

    @property
    def total_batches(self):
        # Note: If len(_txt_files) % _batch_size != 0, this re-uses initial _txt_files
        return int(ceil(float(len(self._txt_files)) /float(self._batch_size)))


def read_data_sets(graph, data_dir, batch_size, numcep, numcontext, thread_count=1):
    # Conditionally download data
    LDC93S1_BASE = "LDC93S1"
    LDC93S1_BASE_URL = "https://catalog.ldc.upenn.edu/desc/addenda/"
    local_file = base.maybe_download(LDC93S1_BASE + ".wav", data_dir, LDC93S1_BASE_URL + LDC93S1_BASE + ".wav")
    _ = base.maybe_download(LDC93S1_BASE + ".txt", data_dir, LDC93S1_BASE_URL + LDC93S1_BASE + ".txt")

    # Create all DataSets, we do not really need separation
    train = dev = test = _read_data_set(graph, data_dir, thread_count, batch_size, numcep, numcontext)

    # Return DataSets
    return DataSets(train, dev, test)

def _read_data_set(graph, data_dir, thread_count, batch_size, numcep, numcontext):
    # Obtain list of txt files
    txt_files = glob(path.join(data_dir, "*.txt"))

    # Return DataSet
    return DataSet(graph, txt_files, thread_count, batch_size, numcep, numcontext)
