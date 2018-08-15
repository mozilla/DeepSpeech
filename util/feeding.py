import pandas
import tensorflow as tf
import itertools
import time
import gc
from tensorflow.python.ops import data_flow_ops

from threading import Thread
from math import ceil
from six.moves import range
from util.audio import audiofile_to_input_vector
from util.gpu import get_available_gpus
from util.text import ctc_label_dense_to_sparse, text_to_char_array

class ModelFeeder(object):
    """
    Feeds data into the model.
    """
    def __init__(self,
                train_set,
                dev_set,
                test_set,
                numcep,
                numcontext,
                alphabet,
                is_train,
                tower_feeder_count=-1,
                rank=0,
                size=0):
        self.train = train_set
        self.dev = dev_set
        self.test = test_set
        self.sets = [train_set, dev_set, test_set]
        self.numcep = numcep
        self.numcontext = numcontext
        self.tower_feeder_count = max(len(get_available_gpus()), 1) if tower_feeder_count < 0 else tower_feeder_count
        self.rank = rank
        self.size = size
        self.is_train = is_train

        self._data_set_index = self.get_data_set_index()
        self._data_set = self.sets[self._data_set_index]


        self.loaders = [_DataSetLoader(self, self._data_set, alphabet, self.rank, self.size, i) for i in range(self.tower_feeder_count)]
        self.iterators = [self.loaders[i]._populate_batch_() for i in range(self.tower_feeder_count)]

    def get_data_set_index(self):
        """
        Get the dataset index.
        """
        if self.is_train:
            return 0
        else:
            return 2

    def next_batch(self, tower_feeder_index):
        '''
        Draw the next batch 
        '''
        self.ph_batch_size = self._data_set.batch_size
        source, source_lengths, target, target_lengths =  self.iterators[tower_feeder_index].get_next()
        sparse_labels = ctc_label_dense_to_sparse(target, target_lengths, self.ph_batch_size)
        return source, source_lengths, sparse_labels

class DataSet(object):
    '''
    Represents a collection of audio samples and their respective transcriptions.
    Takes a set of CSV files producted by importers in /bin.
    '''
    def __init__(self, csvs, batch_size, skip=0, limit=0, ascending=True, next_index=lambda i: i + 1):
        self.batch_size =  batch_size
        self.next_index = next_index
        self.files = None
        for csv in csvs:
            file = pandas.read_csv(csv, encoding='utf-8', engine='python', na_filter=False, error_bad_lines=False)
            if self.files is None:
                self.files = file
            else:
                self.files = self.files.append(file)

        self.files = self.files.sort_values(by="wav_filesize", ascending=ascending).ix[:, ["wav_filename", "transcript"]] \
                         .values[skip:]
        if limit > 0:
            self.files = self.files[:limit]
        self.total_batches = int(ceil(len(self.files) / batch_size))

class _DataSetLoader(object):
    '''
    Internal class that represents an input queue with data from one of the Dataset objects.
    Keeps a DataSet reference to access its samples.
    '''
    def __init__(self, model_feeder, data_set, alphabet, rank, size, i):
        self._model_feeder = model_feeder
        self._data_set = data_set
        self._alphabet = alphabet
        self.epoch_id = 0
        self.rank = rank
        self.size = size
        self.index = (self.epoch_id + self.rank) % self.size

    def generate_batch(self):
        for i in itertools.count(1):
          try:
            file_count = len(self._data_set.files)
            self.index += self._data_set.batch_size
            index = self._data_set.next_index(self.index) % file_count
            wav_file, transcript = self._data_set.files[index]

            source = audiofile_to_input_vector(wav_file, self._model_feeder.numcep, self._model_feeder.numcontext)
            source_len = len(source)
            target = text_to_char_array(transcript, self._alphabet)
            target_len = len(target)
            if source_len < target_len:
                raise ValueError('Error: Audio file {} is too short for transcription.'.format(wav_file))
            yield(source, source_len, target, target_len)
          except:
            self.epoch_id += 1
            self.index = (self.epoch_id + self.rank) % self.size
            continue

    def _populate_batch_(self):
        output_buffer_size = self._data_set.batch_size * 10
        ds = tf.data.Dataset.from_generator(
        self.generate_batch, (tf.float32, tf.int32, tf.int32, tf.int32), (tf.TensorShape([None, None]), tf.TensorShape([]), tf.TensorShape([None,]), tf.TensorShape([])))
        ds = ds.repeat()
        ds = ds.shuffle(output_buffer_size, int(time.time()), reshuffle_each_iteration=True)
        ds = ds.padded_batch(self._data_set.batch_size, 
                                padded_shapes=(tf.TensorShape([None, None]), tf.TensorShape([]), tf.TensorShape([None,]), tf.TensorShape([])))
        ds = ds.prefetch(output_buffer_size)
        ds_iter = ds.make_initializable_iterator()
        return ds_iter

