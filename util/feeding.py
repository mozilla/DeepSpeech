import numpy as np
import tensorflow as tf

from math import ceil
from six.moves import range
from threading import Thread
from util.ctc import ctc_label_dense_to_sparse
from util.gpu import get_available_gpus


class ModelFeeder(object):
    '''
    Feeds data into a model.
    Feeding is parallelized by independent units called tower feeders (usually one per GPU).
    Each tower feeder provides data from runtime switchable sources (train, dev).
    These sources are to be provided by the DataSet instances whose references are kept.
    Creates, owns and delegates to tower_feeder_count internal tower feeder objects.
    '''
    def __init__(self,
                 train_set,
                 dev_set,
                 numcep,
                 numcontext,
                 alphabet,
                 tower_feeder_count=-1,
                 threads_per_queue=4):

        self.train = train_set
        self.dev = dev_set
        self.sets = [train_set, dev_set]
        self.numcep = numcep
        self.numcontext = numcontext
        self.tower_feeder_count = max(len(get_available_gpus()), 1) if tower_feeder_count < 0 else tower_feeder_count
        self.threads_per_queue = threads_per_queue

        self.ph_x = tf.placeholder(tf.float32, [None, 2*numcontext+1, numcep])
        self.ph_x_length = tf.placeholder(tf.int32, [])
        self.ph_y = tf.placeholder(tf.int32, [None,])
        self.ph_y_length = tf.placeholder(tf.int32, [])
        self.ph_batch_size = tf.placeholder(tf.int32, [])
        self.ph_queue_selector = tf.placeholder(tf.int32, name='Queue_Selector')

        self._tower_feeders = [_TowerFeeder(self, i, alphabet) for i in range(self.tower_feeder_count)]

    def start_queue_threads(self, session, coord):
        '''
        Starts required queue threads on all tower feeders.
        '''
        queue_threads = []
        for tower_feeder in self._tower_feeders:
            queue_threads += tower_feeder.start_queue_threads(session, coord)
        return queue_threads

    def close_queues(self, session):
        '''
        Closes queues of all tower feeders.
        '''
        for tower_feeder in self._tower_feeders:
            tower_feeder.close_queues(session)

    def set_data_set(self, feed_dict, data_set):
        '''
        Switches all tower feeders to a different source DataSet.
        The provided feed_dict will get enriched with required placeholder/value pairs.
        The DataSet has to be one of those that got passed into the constructor.
        '''
        index = self.sets.index(data_set)
        assert index >= 0
        feed_dict[self.ph_queue_selector] = index
        feed_dict[self.ph_batch_size] = data_set.batch_size

    def next_batch(self, tower_feeder_index):
        '''
        Draw the next batch from one of the tower feeders.
        '''
        return self._tower_feeders[tower_feeder_index].next_batch()


class DataSet(object):
    '''
    Represents a collection of audio samples and their respective transcriptions.
    Takes a set of CSV files produced by importers in /bin.
    '''
    def __init__(self, data, batch_size, skip=0, limit=0, ascending=True, next_index=lambda i: i + 1):
        self.data = data
        self.data.sort_values(by="features_len", ascending=ascending, inplace=True)
        self.batch_size = batch_size
        self.next_index = next_index
        self.total_batches = int(ceil(len(self.data) / batch_size))


class _DataSetLoader(object):
    '''
    Internal class that represents an input queue with data from one of the DataSet objects.
    Each tower feeder will create and combine three data set loaders to one switchable queue.
    Keeps a ModelFeeder reference for accessing shared settings and placeholders.
    Keeps a DataSet reference to access its samples.
    '''
    def __init__(self, model_feeder, data_set, alphabet):
        self._model_feeder = model_feeder
        self._data_set = data_set
        self.queue = tf.PaddingFIFOQueue(shapes=[[None, 2 * model_feeder.numcontext + 1, model_feeder.numcep], [], [None,], []],
                                                  dtypes=[tf.float32, tf.int32, tf.int32, tf.int32],
                                                  capacity=data_set.batch_size * 8)
        self._enqueue_op = self.queue.enqueue([model_feeder.ph_x, model_feeder.ph_x_length, model_feeder.ph_y, model_feeder.ph_y_length])
        self._close_op = self.queue.close(cancel_pending_enqueues=True)
        self._alphabet = alphabet

    def start_queue_threads(self, session, coord):
        '''
        Starts concurrent queue threads for reading samples from the data set.
        '''
        queue_threads = [Thread(target=self._populate_batch_queue, args=(session, coord))
                         for i in range(self._model_feeder.threads_per_queue)]
        for queue_thread in queue_threads:
            coord.register_thread(queue_thread)
            queue_thread.daemon = True
            queue_thread.start()
        return queue_threads

    def close_queue(self, session):
        '''
        Closes the data set queue.
        '''
        session.run(self._close_op)

    def _populate_batch_queue(self, session, coord):
        '''
        Queue thread routine.
        '''
        file_count = len(self._data_set.data)
        index = -1
        while not coord.should_stop():
            index = self._data_set.next_index(index) % file_count
            features, num_strides, transcript, transcript_len = self._data_set.data.iloc[index]

            # Create a view into the array with overlapping strides of size
            # numcontext (past) + 1 (present) + numcontext (future)
            window_size = 2*self._model_feeder.numcontext+1
            features = np.lib.stride_tricks.as_strided(
                features,
                (num_strides, window_size, self._model_feeder.numcep),
                (features.strides[0], features.strides[0], features.strides[1]),
                writeable=False)

            try:
                session.run(self._enqueue_op, feed_dict={
                    self._model_feeder.ph_x: features,
                    self._model_feeder.ph_x_length: num_strides,
                    self._model_feeder.ph_y: transcript,
                    self._model_feeder.ph_y_length: transcript_len
                })
            except tf.errors.CancelledError:
                return


class _TowerFeeder(object):
    '''
    Internal class that represents a switchable input queue for one tower.
    It creates, owns and combines three _DataSetLoader instances.
    Keeps a ModelFeeder reference for accessing shared settings and placeholders.
    '''
    def __init__(self, model_feeder, index, alphabet):
        self._model_feeder = model_feeder
        self.index = index
        self._loaders = [_DataSetLoader(model_feeder, data_set, alphabet) for data_set in model_feeder.sets]
        self._queues = [set_queue.queue for set_queue in self._loaders]
        self._queue = tf.QueueBase.from_list(model_feeder.ph_queue_selector, self._queues)
        self._close_op = self._queue.close(cancel_pending_enqueues=True)

    def next_batch(self):
        '''
        Draw the next batch from from the combined switchable queue.
        '''
        source, source_lengths, target, target_lengths = self._queue.dequeue_many(self._model_feeder.ph_batch_size)
        sparse_labels = ctc_label_dense_to_sparse(target, target_lengths, self._model_feeder.ph_batch_size)
        return source, source_lengths, sparse_labels

    def start_queue_threads(self, session, coord):
        '''
        Starts the queue threads of all owned _DataSetLoader instances.
        '''
        queue_threads = []
        for set_queue in self._loaders:
            queue_threads += set_queue.start_queue_threads(session, coord)
        return queue_threads

    def close_queues(self, session):
        '''
        Closes queues of all owned _DataSetLoader instances.
        '''
        for set_queue in self._loaders:
            set_queue.close_queue(session)

