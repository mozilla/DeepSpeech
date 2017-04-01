from __future__ import absolute_import, division, print_function

import codecs
import fnmatch
import os
import pandas
import progressbar
import subprocess
import tarfile
import tensorflow as tf
import unicodedata

from glob import glob
from math import ceil
from os import path
from six.moves import range
from sox import Transformer
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.platform import gfile
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
        self._y = tf.placeholder(tf.int32, [None, ])
        self._y_length = tf.placeholder(tf.int32, [])
        self.example_queue = tf.PaddingFIFOQueue(shapes=[[None, numcep + (2 * numcep * numcontext)], [], [None, ], []],
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
    # Check if we can convert FLAC with SoX before we start
    sox_help_out = subprocess.check_output(["sox", "-h"]).decode('latin-1')
    if sox_help_out.find("flac") == -1:
        print("Error: SoX doesn't support FLAC. Please install SoX with FLAC support and try again.")
        exit(1)

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
        train_files, dev_files, test_files = _download_and_preprocess_data(data_dir)
        print("Finished pre-processing librivox. Initializing dataset...")

    # Create train DataSet from all the train archives
    train = None
    if "train" in sets:
        train = _read_data_set(train_files, thread_count, train_batch_size, numcep, numcontext, stride=stride, offset=offset, next_index=lambda i: next_index('train', i), limit=limit_train)

    # Create dev DataSet from all the dev archives
    dev = None
    if "dev" in sets:
        dev = _read_data_set(dev_files, thread_count, dev_batch_size, numcep, numcontext, stride=stride, offset=offset, next_index=lambda i: next_index('dev', i), limit=limit_dev)

    # Create test DataSet from all the test archives
    test = None
    if "test" in sets:
        test = _read_data_set(test_files, thread_count, test_batch_size, numcep, numcontext, stride=stride, offset=offset, next_index=lambda i: next_index('test', i), limit=limit_test)

    # Return DataSets
    return DataSets(train, dev, test)

def _download_and_preprocess_data(data_dir):
    # Conditionally download data to data_dir
    print("Downloading Librivox data set (55GB) into {} if not already present...".format(data_dir))
    with progressbar.ProgressBar(max_value=7, widget=progressbar.AdaptiveETA) as bar:
        TRAIN_CLEAN_100_URL = "http://www.openslr.org/resources/12/train-clean-100.tar.gz"
        TRAIN_CLEAN_360_URL = "http://www.openslr.org/resources/12/train-clean-360.tar.gz"
        TRAIN_OTHER_500_URL = "http://www.openslr.org/resources/12/train-other-500.tar.gz"

        DEV_CLEAN_URL = "http://www.openslr.org/resources/12/dev-clean.tar.gz"
        DEV_OTHER_URL = "http://www.openslr.org/resources/12/dev-other.tar.gz"

        TEST_CLEAN_URL = "http://www.openslr.org/resources/12/test-clean.tar.gz"
        TEST_OTHER_URL = "http://www.openslr.org/resources/12/test-other.tar.gz"

        def filename_of(x): return path.split(x)[1]
        train_clean_100 = base.maybe_download(filename_of(TRAIN_CLEAN_100_URL), data_dir, TRAIN_CLEAN_100_URL)
        bar.update(0)
        train_clean_360 = base.maybe_download(filename_of(TRAIN_CLEAN_360_URL), data_dir, TRAIN_CLEAN_360_URL)
        bar.update(1)
        train_other_500 = base.maybe_download(filename_of(TRAIN_OTHER_500_URL), data_dir, TRAIN_OTHER_500_URL)
        bar.update(2)

        dev_clean = base.maybe_download(filename_of(DEV_CLEAN_URL), data_dir, DEV_CLEAN_URL)
        bar.update(3)
        dev_other = base.maybe_download(filename_of(DEV_OTHER_URL), data_dir, DEV_OTHER_URL)
        bar.update(4)

        test_clean = base.maybe_download(filename_of(TEST_CLEAN_URL), data_dir, TEST_CLEAN_URL)
        bar.update(5)
        test_other = base.maybe_download(filename_of(TEST_OTHER_URL), data_dir, TEST_OTHER_URL)
        bar.update(6)

    # Conditionally extract LibriSpeech data
    # We extract each archive into data_dir, but test for existence in
    # data_dir/LibriSpeech because the archives share that root.
    print("Extracting librivox data if not already extracted...")
    with progressbar.ProgressBar(max_value=7, widget=progressbar.AdaptiveETA) as bar:
        LIBRIVOX_DIR = "LibriSpeech"
        work_dir = os.path.join(data_dir, LIBRIVOX_DIR)

        _maybe_extract(data_dir, os.path.join(LIBRIVOX_DIR, "train-clean-100"), train_clean_100)
        bar.update(0)
        _maybe_extract(data_dir, os.path.join(LIBRIVOX_DIR, "train-clean-360"), train_clean_360)
        bar.update(1)
        _maybe_extract(data_dir, os.path.join(LIBRIVOX_DIR, "train-other-500"), train_other_500)
        bar.update(2)

        _maybe_extract(data_dir, os.path.join(LIBRIVOX_DIR, "dev-clean"), dev_clean)
        bar.update(3)
        _maybe_extract(data_dir, os.path.join(LIBRIVOX_DIR, "dev-other"), dev_other)
        bar.update(4)

        _maybe_extract(data_dir, os.path.join(LIBRIVOX_DIR, "test-clean"), test_clean)
        bar.update(5)
        _maybe_extract(data_dir, os.path.join(LIBRIVOX_DIR, "test-other"), test_other)
        bar.update(6)

    # Convert FLAC data to wav, from:
    #  data_dir/LibriSpeech/split/1/2/1-2-3.flac
    # to:
    #  data_dir/LibriSpeech/split-wav/1-2-3.wav
    #
    # And split LibriSpeech transcriptions, from:
    #  data_dir/LibriSpeech/split/1/2/1-2.trans.txt
    # to:
    #  data_dir/LibriSpeech/split-wav/1-2-0.txt
    #  data_dir/LibriSpeech/split-wav/1-2-1.txt
    #  data_dir/LibriSpeech/split-wav/1-2-2.txt
    #  ...
    print("Converting FLAC to WAV and splitting transcriptions...")
    with progressbar.ProgressBar(max_value=7,  widget=progressbar.AdaptiveETA) as bar:
        train_100 = _convert_audio_and_split_sentences(work_dir, "train-clean-100", "train-clean-100-wav")
        bar.update(0)
        train_360 = _convert_audio_and_split_sentences(work_dir, "train-clean-360", "train-clean-360-wav")
        bar.update(1)
        train_500 = _convert_audio_and_split_sentences(work_dir, "train-other-500", "train-other-500-wav")
        bar.update(2)

        dev_clean = _convert_audio_and_split_sentences(work_dir, "dev-clean", "dev-clean-wav")
        bar.update(3)
        dev_other = _convert_audio_and_split_sentences(work_dir, "dev-other", "dev-other-wav")
        bar.update(4)

        test_clean = _convert_audio_and_split_sentences(work_dir, "test-clean", "test-clean-wav")
        bar.update(5)
        test_other = _convert_audio_and_split_sentences(work_dir, "test-other", "test-other-wav")
        bar.update(6)

    # Write sets to disk as CSV files
    train_100.to_csv(os.path.join(data_dir, "librivox-train-clean-100.csv"), index=False)
    train_360.to_csv(os.path.join(data_dir, "librivox-train-clean-360.csv"), index=False)
    train_500.to_csv(os.path.join(data_dir, "librivox-train-other-500.csv"), index=False)

    dev_clean.to_csv(os.path.join(data_dir, "librivox-dev-clean.csv"), index=False)
    dev_other.to_csv(os.path.join(data_dir, "librivox-dev-other.csv"), index=False)

    test_clean.to_csv(os.path.join(data_dir, "librivox-test-clean.csv"), index=False)
    test_other.to_csv(os.path.join(data_dir, "librivox-test-other.csv"), index=False)

def _maybe_extract(data_dir, extracted_data, archive):
    # If data_dir/extracted_data does not exist, extract archive in data_dir
    if not gfile.Exists(os.path.join(data_dir, extracted_data)):
        tar = tarfile.open(archive)
        tar.extractall(data_dir)
        tar.close()

def _convert_audio_and_split_sentences(extracted_dir, data_set, dest_dir):
    source_dir = os.path.join(extracted_dir, data_set)
    target_dir = os.path.join(extracted_dir, dest_dir)

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Loop over transcription files and split each one
    #
    # The format for each file 1-2.trans.txt is:
    #  1-2-0 transcription of 1-2-0.flac
    #  1-2-1 transcription of 1-2-1.flac
    #  ...
    #
    # Each file is then split into several files:
    #  1-2-0.txt (contains transcription of 1-2-0.flac)
    #  1-2-1.txt (contains transcription of 1-2-1.flac)
    #  ...
    #
    # We also convert the corresponding FLACs to WAV in the same pass
    files = []
    for root, dirnames, filenames in os.walk(source_dir):
        for filename in fnmatch.filter(filenames, '*.trans.txt'):
            trans_filename = os.path.join(root, filename)
            with codecs.open(trans_filename, "r", "utf-8") as fin:
                for line in fin:
                    # Parse each segment line
                    first_space = line.find(" ")
                    seqid, transcript = line[:first_space], line[first_space+1:]

                    transcript = unicodedata.normalize("NFKD", transcript)  \
                                            .encode("ascii", "ignore")      \
                                            .decode("ascii", "ignore")

                    transcript = transcript.lower().strip()

                    # Convert corresponding FLAC to a WAV
                    flac_file = os.path.join(root, seqid + ".flac")
                    wav_file = os.path.join(target_dir, seqid + ".wav")
                    if not os.path.exists(wav_file):
                        Transformer().build(flac_file, wav_file)
                    wav_filesize = os.path.getsize(wav_file)

                    files.append((wav_file, wav_filesize, transcript))

    return pandas.DataFrame(data=files, columns=["wav_filename", "wav_filesize", "transcript"])

def _read_data_set(filelist, thread_count, batch_size, numcep, numcontext, stride=1, offset=0, next_index=lambda i: i + 1, limit=0):
    # Optionally apply dataset size limit
    if limit > 0:
        filelist = filelist.iloc[:limit]

    filelist = filelist[offset::stride]

    # Return DataSet
    return DataSet(txt_files, thread_count, batch_size, numcep, numcontext, next_index=next_index)
