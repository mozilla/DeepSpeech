import wave
import random
import tarfile
import threading
import numpy as np
import tensorflow as tf
import unicodedata
import codecs

from os import path
from os import rmdir
from os import remove
from glob import glob
from math import ceil
from Queue import Queue
from os import makedirs
from sox import Transformer
from itertools import cycle
from os.path import getsize
from threading import Thread
from Queue import PriorityQueue
from util.stm import parse_stm_file
from util.gpu import get_available_gpus
from util.text import text_to_char_array, ctc_label_dense_to_sparse
from tensorflow.python.platform import gfile
from util.audio import audiofile_to_input_vector
from tensorflow.contrib.learn.python.learn.datasets import base

class DataSets(object):
    def __init__(self, train, dev, test):
        self._dev = dev
        self._test = test
        self._train = train

    def start_queue_threads(self, session):
        self._dev.start_queue_threads(session)
        self._test.start_queue_threads(session)
        self._train.start_queue_threads(session)

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
        self._numcep = numcep
        self._x = tf.placeholder(tf.float32, [None, numcep + (2 * numcep * numcontext)])
        self._x_length = tf.placeholder(tf.int32, [])
        self._y = tf.placeholder(tf.int32, [None,])
        self._y_length = tf.placeholder(tf.int32, [])
        self._example_queue = tf.PaddingFIFOQueue(shapes=[[None, numcep + (2 * numcep * numcontext)], [], [None,], []],
                                                  dtypes=[tf.float32, tf.int32, tf.int32, tf.int32],
                                                  capacity=2 * self._get_device_count() * batch_size)
        self._enqueue_op = self._example_queue.enqueue([self._x, self._x_length, self._y, self._y_length])
        self._txt_files = txt_files
        self._batch_size = batch_size
        self._numcontext = numcontext
        self._thread_count = thread_count
        self._files_circular_list = self._create_files_circular_list()

    def _get_device_count(self):
        available_gpus = get_available_gpus()
        return max(len(available_gpus), 1)

    def start_queue_threads(self, session):
        batch_threads = [Thread(target=self._populate_batch_queue, args=(session,)) for i in xrange(self._thread_count)]
        for batch_thread in batch_threads:
            batch_thread.daemon = True
            batch_thread.start()
        return batch_threads

    def _create_files_circular_list(self):
        priorityQueue = PriorityQueue()
        for txt_file in self._txt_files:
          stm_dir = path.sep + "stm" + path.sep
          wav_dir = path.sep + "wav" + path.sep
          wav_file = path.splitext(txt_file.replace(stm_dir, wav_dir))[0] + ".wav"
          wav_file_size = getsize(wav_file)
          priorityQueue.put((wav_file_size, (txt_file, wav_file)))
        files_list = []
        while not priorityQueue.empty():
            priority, (txt_file, wav_file) = priorityQueue.get()
            files_list.append((txt_file, wav_file))
        return cycle(files_list)

    def _populate_batch_queue(self, session):
        for txt_file, wav_file in self._files_circular_list:
            source = audiofile_to_input_vector(wav_file, self._numcep, self._numcontext)
            source_len = len(source)
            with codecs.open(txt_file, encoding="utf-8") as open_txt_file:
                target = unicodedata.normalize("NFKD", open_txt_file.read()).encode("ascii", "ignore")
                target = text_to_char_array(target)
            target_len = len(target)
            try:
                session.run(self._enqueue_op, feed_dict={
                    self._x: source,
                    self._x_length: source_len,
                    self._y: target,
                    self._y_length: target_len})
            except (RuntimeError, tf.errors.CancelledError):
                return

    def next_batch(self):
        source, source_lengths, target, target_lengths = self._example_queue.dequeue_many(self._batch_size)
        sparse_labels = ctc_label_dense_to_sparse(target, target_lengths, self._batch_size)
        return source, source_lengths, sparse_labels

    @property
    def total_batches(self):
        # Note: If len(_txt_files) % _batch_size != 0, this re-uses initial _txt_files
        return int(ceil(float(len(self._txt_files)) /float(self._batch_size)))


def read_data_sets(data_dir, batch_size, numcep, numcontext, thread_count=8, limit_dev=0, limit_test=0, limit_train=0):
    # Conditionally download data
    TED_DATA = "TEDLIUM_release2.tar.gz"
    TED_DATA_URL = "http://www.openslr.org/resources/19/TEDLIUM_release2.tar.gz"
    local_file = base.maybe_download(TED_DATA, data_dir, TED_DATA_URL)

    # Conditionally extract TED data
    TED_DIR = "TEDLIUM_release2"
    _maybe_extract(data_dir, TED_DIR, local_file)

    # Conditionally convert TED sph data to wav
    _maybe_convert_wav(data_dir, TED_DIR)

    # Conditionally split TED wav data
    _maybe_split_wav(data_dir, TED_DIR)

    # Conditionally split TED stm data
    _maybe_split_stm(data_dir, TED_DIR)

    # Create dev DataSet
    dev = _read_data_set(data_dir, TED_DIR, "dev", thread_count, batch_size, numcep, numcontext, limit=limit_dev)

    # Create test DataSet
    test = _read_data_set(data_dir, TED_DIR, "test", thread_count, batch_size, numcep, numcontext, limit=limit_test)

    # Create train DataSet
    train = _read_data_set(data_dir, TED_DIR, "train", thread_count, batch_size, numcep, numcontext, limit=limit_train)

    # Return DataSets
    return DataSets(train, dev, test)

def _maybe_extract(data_dir, extracted_data, archive):
    # If data_dir/extracted_data does not exist, extract archive in data_dir
    if not gfile.Exists(path.join(data_dir, extracted_data)):
      tar = tarfile.open(archive)
      tar.extractall(data_dir)
      tar.close()

def _maybe_convert_wav(data_dir, extracted_data):
    # Create extracted_data dir
    extracted_dir = path.join(data_dir, extracted_data)

    # Conditionally convert dev sph to wav
    _maybe_convert_wav_dataset(extracted_dir, "dev")

    # Conditionally convert train sph to wav
    _maybe_convert_wav_dataset(extracted_dir, "train")

    # Conditionally convert test sph to wav
    _maybe_convert_wav_dataset(extracted_dir, "test")

def _maybe_convert_wav_dataset(extracted_dir, data_set):
    # Create source dir
    source_dir = path.join(extracted_dir, data_set, "sph")

    # Create target dir
    target_dir = path.join(extracted_dir, data_set, "wav")

    # Conditionally convert sph files to wav files
    if not gfile.Exists(target_dir):
        # Create target_dir
        makedirs(target_dir)

        # Loop over sph files in source_dir and convert each to wav
        for sph_file in glob(path.join(source_dir, "*.sph")):
            transformer = Transformer()
            wav_filename = path.splitext(path.basename(sph_file))[0] + ".wav"
            wav_file = path.join(target_dir, wav_filename)
            transformer.build(sph_file, wav_file)
            remove(sph_file)

        # Remove source_dir
        rmdir(source_dir)

def _maybe_split_wav(data_dir, extracted_data):
    # Create extracted_data dir
    extracted_dir = path.join(data_dir, extracted_data)

    # Conditionally split dev wav
    _maybe_split_wav_dataset(extracted_dir, "dev")

    # Conditionally split train wav
    _maybe_split_wav_dataset(extracted_dir, "train")

    # Conditionally split test wav
    _maybe_split_wav_dataset(extracted_dir, "test")

def _maybe_split_wav_dataset(extracted_dir, data_set):
    # Create stm dir
    stm_dir = path.join(extracted_dir, data_set, "stm")

    # Create wav dir
    wav_dir = path.join(extracted_dir, data_set, "wav")

    # Loop over stm files and split corresponding wav
    for stm_file in glob(path.join(stm_dir, "*.stm")):
        # Parse stm file
        stm_segments = parse_stm_file(stm_file)

        # Open wav corresponding to stm_file
        wav_filename = path.splitext(path.basename(stm_file))[0] + ".wav"
        wav_file = path.join(wav_dir, wav_filename)
        origAudio = wave.open(wav_file,'r')

        # Loop over stm_segments and split wav_file for each segment
        for stm_segment in stm_segments:
            # Create wav segment filename
            start_time = stm_segment.start_time
            stop_time = stm_segment.stop_time
            new_wav_filename = path.splitext(path.basename(stm_file))[0] + "-" + str(start_time) + "-" + str(stop_time) + ".wav"
            new_wav_file = path.join(wav_dir, new_wav_filename)

            # If the wav segment filename does not exist create it
            if not gfile.Exists(new_wav_file):
                _split_wav(origAudio, start_time, stop_time, new_wav_file)

        # Close origAudio
        origAudio.close()

        # Remove wav_file
        remove(wav_file)

def _split_wav(origAudio, start_time, stop_time, new_wav_file):
    frameRate = origAudio.getframerate()
    origAudio.setpos(int(start_time*frameRate))
    chunkData = origAudio.readframes(int((stop_time - start_time)*frameRate))
    chunkAudio = wave.open(new_wav_file,'w')
    chunkAudio.setnchannels(origAudio.getnchannels())
    chunkAudio.setsampwidth(origAudio.getsampwidth())
    chunkAudio.setframerate(frameRate)
    chunkAudio.writeframes(chunkData)
    chunkAudio.close()

def _maybe_split_stm(data_dir, extracted_data):
    # Create extracted_data dir
    extracted_dir = path.join(data_dir, extracted_data)

    # Conditionally split dev stm
    _maybe_split_stm_dataset(extracted_dir, "dev")

    # Conditionally split train stm
    _maybe_split_stm_dataset(extracted_dir, "train")

    # Conditionally split test stm
    _maybe_split_stm_dataset(extracted_dir, "test")

def _maybe_split_stm_dataset(extracted_dir, data_set):
    # Create stm dir
    stm_dir = path.join(extracted_dir, data_set, "stm")

    # Obtain stm files
    stm_files = glob(path.join(stm_dir, "*.stm"))

    # Loop over stm files and split each one
    for stm_file in stm_files:
        # Parse stm file
        stm_segments = parse_stm_file(stm_file)

        # Loop over stm_segments and create txt file for each one
        for stm_segment in stm_segments:
            start_time = stm_segment.start_time
            stop_time = stm_segment.stop_time
            txt_filename = path.splitext(path.basename(stm_file))[0] + "-" + str(start_time) + "-" + str(stop_time) + ".txt"
            txt_file = path.join(stm_dir, txt_filename)

            # If the txt segment file does not exist create it
            if not gfile.Exists(txt_file):
                with open(txt_file, "w+") as f:
                  f.write(stm_segment.transcript)

        # Remove stm_file
        remove(stm_file)

def _read_data_set(data_dir, extracted_data, data_set, thread_count, batch_size, numcep, numcontext, limit=0):
    # Create stm dir
    stm_dir = path.join(data_dir, extracted_data, data_set, "stm")

    # Obtain list of txt files
    txt_files = glob(path.join(stm_dir, "*.txt"))
    if limit > 0:
        txt_files = txt_files[:limit]

    # Return DataSet
    return DataSet(txt_files, thread_count, batch_size, numcep, numcontext)
