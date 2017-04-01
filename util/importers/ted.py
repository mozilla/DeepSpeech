from __future__ import absolute_import, division, print_function

import codecs
import pandas
import tarfile
import tensorflow as tf
import unicodedata
import wave

from glob import glob
from math import ceil
from os import makedirs, path, remove, rmdir
from six.moves import range
from sox import Transformer
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.platform import gfile
from threading import Thread
from util.audio import audiofile_to_input_vector
from util.data_set_helpers import DataSets
from util.gpu import get_available_gpus
from util.stm import parse_stm_file
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
        TED_DATA = "TEDLIUM_release2.tar.gz"
        TED_DATA_URL = "http://www.openslr.org/resources/19/TEDLIUM_release2.tar.gz"
        local_file = base.maybe_download(TED_DATA, data_dir, TED_DATA_URL)

        # Conditionally extract TED data
        TED_DIR = "TEDLIUM_release2"
        _maybe_extract(data_dir, TED_DIR, local_file)

        # Conditionally convert TED sph data to wav
        _maybe_convert_wav(data_dir, TED_DIR)

        # Conditionally split TED wav and text data into sentences
        train_files, dev_files, test_files = _maybe_split_sentences(data_dir, TED_DIR)

        # Write sets to disk as CSV files
        train_files.to_csv(path.join(data_dir, "ted-train.csv"), index=False)
        dev_files.to_csv(path.join(data_dir, "ted-dev.csv"), index=False)
        test_files.to_csv(path.join(data_dir, "ted-test.csv"), index=False)

    # Create dev DataSet
    dev = None
    if "dev" in sets:
        dev = _read_data_set(train_files, thread_count, dev_batch_size, numcep, numcontext, stride=stride, offset=offset, next_index=lambda i: next_index('dev', i), limit=limit_dev)

    # Create test DataSet
    test = None
    if "test" in sets:
        test = _read_data_set(test_files, thread_count, test_batch_size, numcep, numcontext, stride=stride, offset=offset, next_index=lambda i: next_index('test', i), limit=limit_test)

    # Create train DataSet
    train = None
    if "train" in sets:
        train = _read_data_set(dev_files, thread_count, train_batch_size, numcep, numcontext, stride=stride, offset=offset, next_index=lambda i: next_index('train', i), limit=limit_train)

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

def _maybe_split_sentences(data_dir, extracted_data):
    # Create extracted_data dir
    extracted_dir = path.join(data_dir, extracted_data)

    # Conditionally split dev wav
    dev_files = _maybe_split_dataset(extracted_dir, "dev")

    # Conditionally split train wav
    train_files = _maybe_split_dataset(extracted_dir, "train")

    # Conditionally split test wav
    test_files = _maybe_split_dataset(extracted_dir, "test")

    return train_files, dev_files, test_files

def _maybe_split_dataset(extracted_dir, data_set):
    # Create stm dir
    stm_dir = path.join(extracted_dir, data_set, "stm")

    # Create wav dir
    wav_dir = path.join(extracted_dir, data_set, "wav")

    files = []

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

            new_wav_filesize = path.getsize(new_wav_file)
            files.append((new_wav_file, new_wav_filesize, stm_segment.transcript))

        # Close origAudio
        origAudio.close()

        # Remove wav_file
        remove(wav_file)

    return pandas.DataFrame(data=files, columns=["wav_filename", "wav_filesize", "transcript"])

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

def _read_data_set(filelist, thread_count, batch_size, numcep, numcontext, stride=1, offset=0, next_index=lambda i: i + 1, limit=0):
    # Optionally apply dataset size limit
    if limit > 0:
        filelist = filelist.iloc[:limit]

    filelist = filelist[offset::stride]

    # Return DataSet
    return DataSet(filelist, thread_count, batch_size, numcep, numcontext, next_index=next_index)
