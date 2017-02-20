
import threading
import numpy as np
import tensorflow as tf

from util.text import ctc_label_dense_to_sparse

class DataSets(object):
    def __init__(self, train, dev, test):
        '''Container for train, dev and test sets of one corpus.

        Args:
            train (DataSet): the train data set of the corpus
            dev (DataSet): the validation data set of the corpus
            test (DataSet): the test data set of the corpus
        '''
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

class SwitchableDataSet(object):
    def __init__(self, data_sets):
        '''Data set that is wrapping a data sets instance to switch between train, dev and test instances during training.

        Args:
            data_sets (DataSets): the corpus container holding all three data sets
        '''
        self._data_sets = data_sets
        self._sets = [data_sets.train, data_sets.dev, data_sets.test]
        self._queues = [s._example_queue for s in self._sets]
        self._queue_selector = tf.placeholder(tf.int32, name='Queue_Selector')
        self._queue = tf.QueueBase.from_list(self._queue_selector, self._queues)
        self._close_op = self._queue.close(cancel_pending_enqueues=True)
        self._data_set = data_sets.train

    def set_data_set(self, feed_dict, data_set):
        index = self._sets.index(data_set)
        assert index >= 0
        feed_dict[self._queue_selector] = index
        self._data_set = data_set

    def start_queue_threads(self, session, coord):
        batch_threads = []
        for s in self._sets:
            batch_threads += s.start_queue_threads(session, coord)
        return batch_threads

    def close_queue(self, session):
        session.run(self._close_op, feed_dict={ self._queue_selector: 0 })
        for s in self._sets:
            s.close_queue(session)

    def next_batch(self):
        source, source_lengths, target, target_lengths = self._queue.dequeue_many(self._data_set.batch_size)
        sparse_labels = ctc_label_dense_to_sparse(target, target_lengths, self._data_set.batch_size)
        return source, source_lengths, sparse_labels
