import fnmatch
import numpy as np
import os
import random
import subprocess
import tarfile

from glob import glob
from itertools import cycle
from math import ceil
from sox import Transformer
from Queue import PriorityQueue
from Queue import Queue
from shutil import rmtree
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.platform import gfile
from threading import Thread
from util.audio import audiofile_to_input_vector
from util.gpu import get_available_gpus
from util.text import texts_to_sparse_tensor

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
        self._files_circular_list = self._create_files_circular_list()
        self._start_queue_threads()
    
    def _get_device_count(self):
        available_gpus = get_available_gpus()
        return max(len(available_gpus), 1)
    
    def _start_queue_threads(self):
        batch_threads = [Thread(target=self._populate_batch_queue) for i in xrange(self._thread_count)]
        for batch_thread in batch_threads:
            batch_thread.daemon = True
            batch_thread.start()
    
    def _create_files_circular_list(self):
        priorityQueue = PriorityQueue()
        for txt_file in self._txt_files:
          wav_file = os.path.splitext(txt_file)[0] + ".wav"
          wav_file_size = os.path.getsize(wav_file)
          priorityQueue.put((wav_file_size, (txt_file, wav_file)))
        files_list = []
        while not priorityQueue.empty():
            priority, (txt_file, wav_file) = priorityQueue.get()
            files_list.append((txt_file, wav_file))
        return cycle(files_list)
    
    def _populate_batch_queue(self):
        with self._graph.as_default():
            while True:
                n_steps = 0
                sources = []
                targets = []
                for index, (txt_file, wav_file) in enumerate(self._files_circular_list):
                    if index >= self._batch_size:
                        break
                    next_source = audiofile_to_input_vector(wav_file, self._numcep, self._numcontext)
                    if n_steps < next_source.shape[0]:
                        n_steps = next_source.shape[0]
                    sources.append(next_source)
                    with open(txt_file) as open_txt_file:
                        targets.append(open_txt_file.read())
                target = texts_to_sparse_tensor(targets)
                for index, next_source in enumerate(sources):
                    npad = ((0,(n_steps - next_source.shape[0])), (0,0))
                    sources[index] = np.pad(next_source, pad_width=npad, mode='constant')
                source = np.array(sources)
                self._batch_queue.put((source, target))
    
    def next_batch(self):
        source, target = self._batch_queue.get()
        return (source, target)
    
    @property
    def total_batches(self):
        # Note: If len(_txt_files) % _batch_size != 0, this re-uses initial _txt_files
        return int(ceil(float(len(self._txt_files)) /float(self._batch_size)))


def read_data_sets(graph, data_dir, batch_size, numcep, numcontext, thread_count=8):
    # Check if we can convert FLAC with SoX before we start
    sox_help_out = subprocess.check_output(["sox", "-h"])
    if sox_help_out.find("flac") == -1:
        print("Error: SoX doesn't support FLAC. Please install SoX with FLAC support and try again.")
        exit(1)
    
    # Conditionally download data to data_dir
    TRAIN_CLEAN_100_URL = "http://www.openslr.org/resources/12/train-clean-100.tar.gz"
    TRAIN_CLEAN_360_URL = "http://www.openslr.org/resources/12/train-clean-360.tar.gz"
    TRAIN_OTHER_500_URL = "http://www.openslr.org/resources/12/train-other-500.tar.gz"
    
    DEV_CLEAN_URL = "http://www.openslr.org/resources/12/dev-clean.tar.gz"
    DEV_OTHER_URL = "http://www.openslr.org/resources/12/dev-other.tar.gz"
    
    TEST_CLEAN_URL = "http://www.openslr.org/resources/12/test-clean.tar.gz"
    TEST_OTHER_URL = "http://www.openslr.org/resources/12/test-other.tar.gz"
    
    train_clean_100 = base.maybe_download("train-clean-100.tar.gz", data_dir, TRAIN_CLEAN_100_URL)
    train_clean_360 = base.maybe_download("train-clean-360.tar.gz", data_dir, TRAIN_CLEAN_360_URL)
    train_other_500 = base.maybe_download("train-other-500.tar.gz", data_dir, TRAIN_OTHER_500_URL)
    
    dev_clean = base.maybe_download("dev-clean.tar.gz", data_dir, DEV_CLEAN_URL)
    dev_other = base.maybe_download("dev-other.tar.gz", data_dir, DEV_OTHER_URL)
    
    test_clean = base.maybe_download("test-clean.tar.gz", data_dir, TEST_CLEAN_URL)
    test_other = base.maybe_download("test-other.tar.gz", data_dir, TEST_OTHER_URL)
    
    # Conditionally extract LibriSpeech data
    # We extract each archive into data_dir, but test for existence in
    # data_dir/LibriSpeech because the archives share that root.
    LIBRIVOX_DIR = "LibriSpeech"
    work_dir = os.path.join(data_dir, LIBRIVOX_DIR)
    
    _maybe_extract(data_dir, os.path.join(LIBRIVOX_DIR, "train-clean-100"), train_clean_100)
    _maybe_extract(data_dir, os.path.join(LIBRIVOX_DIR, "train-clean-360"), train_clean_360)
    _maybe_extract(data_dir, os.path.join(LIBRIVOX_DIR, "train-other-500"), train_other_500)
    
    _maybe_extract(data_dir, os.path.join(LIBRIVOX_DIR, "dev-clean"), dev_clean)
    _maybe_extract(data_dir, os.path.join(LIBRIVOX_DIR, "dev-other"), dev_other)
    
    _maybe_extract(data_dir, os.path.join(LIBRIVOX_DIR, "test-clean"), test_clean)
    _maybe_extract(data_dir, os.path.join(LIBRIVOX_DIR, "test-other"), test_other)
    
    # Conditionally convert FLAC data to wav, from:
    #  data_dir/LibriSpeech/split/1/2/1-2-3.flac
    # to:
    #  data_dir/LibriSpeech/split-wav/1-2-3.wav
    _maybe_convert_wav(work_dir, "train-clean-100", "train-clean-100-wav")
    _maybe_convert_wav(work_dir, "train-clean-360", "train-clean-360-wav")
    _maybe_convert_wav(work_dir, "train-other-500", "train-other-500-wav")
    
    _maybe_convert_wav(work_dir, "dev-clean", "dev-clean-wav")
    _maybe_convert_wav(work_dir, "dev-other", "dev-other-wav")
    
    _maybe_convert_wav(work_dir, "test-clean", "test-clean-wav")
    _maybe_convert_wav(work_dir, "test-other", "test-other-wav")
    
    # Conditionally split LibriSpeech transcriptions, from:
    #  data_dir/LibriSpeech/split/1/2/1-2.trans.txt
    # to:
    #  data_dir/LibriSpeech/split-wav/1-2-0.txt
    #  data_dir/LibriSpeech/split-wav/1-2-1.txt
    #  data_dir/LibriSpeech/split-wav/1-2-2.txt
    #  ...
    _maybe_split_transcriptions(work_dir, "train-clean-100", "train-clean-100-wav")
    _maybe_split_transcriptions(work_dir, "train-clean-360", "train-clean-360-wav")
    _maybe_split_transcriptions(work_dir, "train-other-500", "train-other-500-wav")
    
    _maybe_split_transcriptions(work_dir, "dev-clean", "dev-clean-wav")
    _maybe_split_transcriptions(work_dir, "dev-other", "dev-other-wav")
    
    _maybe_split_transcriptions(work_dir, "test-clean", "test-clean-wav")
    _maybe_split_transcriptions(work_dir, "test-other", "test-other-wav")
    
    # Create train DataSet from all the train archives
    train = _read_data_set(graph, work_dir, "train-*-wav", thread_count, batch_size, numcep, numcontext)
    
    # Create dev DataSet from all the dev archives
    dev = _read_data_set(graph, work_dir, "dev-*-wav", thread_count, batch_size, numcep, numcontext)
    
    # Create test DataSet from all the test archives
    test = _read_data_set(graph, work_dir, "test-*-wav", thread_count, batch_size, numcep, numcontext)
    
    # Return DataSets
    return DataSets(train, dev, test)

def _maybe_extract(data_dir, extracted_data, archive):
    # If data_dir/extracted_data does not exist, extract archive in data_dir
    if not gfile.Exists(os.path.join(data_dir, extracted_data)):
        tar = tarfile.open(archive)
        tar.extractall(data_dir)
        tar.close()
        # os.remove(archive)

def _maybe_convert_wav(data_dir, extracted_data, converted_data):
    source_dir = os.path.join(data_dir, extracted_data)
    target_dir = os.path.join(data_dir, converted_data)
    
    # Conditionally convert FLAC files to wav files
    if not gfile.Exists(target_dir):
        # Create target_dir
        os.makedirs(target_dir)
        
        # Loop over FLAC files in source_dir and convert each to wav
        for root, dirnames, filenames in os.walk(source_dir):
            for filename in fnmatch.filter(filenames, '*.flac'):
                flac_file = os.path.join(root, filename)
                wav_filename = os.path.splitext(os.path.basename(flac_file))[0] + ".wav"
                wav_file = os.path.join(target_dir, wav_filename)
                transformer = Transformer()
                transformer.build(flac_file, wav_file)
                os.remove(flac_file)

def _maybe_split_transcriptions(extracted_dir, data_set, dest_dir):
    source_dir = os.path.join(extracted_dir, data_set)
    target_dir = os.path.join(extracted_dir, dest_dir)
    
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
    for root, dirnames, filenames in os.walk(source_dir):
        for filename in fnmatch.filter(filenames, '*.trans.txt'):
            trans_filename = os.path.join(root, filename)
            with open(trans_filename, "r") as fin:
                for line in fin:
                    first_space = line.find(" ")
                    txt_file = line[:first_space] + ".txt"
                    with open(os.path.join(target_dir, txt_file), "w") as fout:
                        fout.write(line[first_space+1:].lower().strip("\n"))
            os.remove(trans_filename)

def _read_data_set(graph, work_dir, data_set, thread_count, batch_size, numcep, numcontext):
    # Create data set dir
    dataset_dir = os.path.join(work_dir, data_set)
    
    # Obtain list of txt files
    txt_files = glob(os.path.join(dataset_dir, "*.txt"))
    
    # Return DataSet
    return DataSet(graph, txt_files, thread_count, batch_size, numcep, numcontext)
