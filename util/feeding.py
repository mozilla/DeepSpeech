import pandas
import tensorflow as tf
import time

from threading import Thread, Lock
from math import ceil
from six.moves import range
from util.audio import audiofile_to_input_vector
from util.gpu import get_available_gpus
from util.text import ctc_label_dense_to_sparse, text_to_char_array

class ModelFeeder(object):
    '''
    Feeds data into a model.
    Feeding is parallelized by independent units called tower feeders (usually one per GPU).
    Each tower feeder provides data from three runtime switchable sources (train, dev, test).
    These sources are to be provided by three DataSet instances who's references are kept.
    Creates, owns and delegates to tower_feeder_count internal tower feeder objects.
    '''
    def __init__(self,
                 train_set,
                 dev_set,
                 test_set,
                 numcep,
                 numcontext,
                 tower_feeder_count=-1,
                 threads_per_queue=1,
                 queue_capacity=500):

        self.len_threshold = 9000
        self.train = train_set
        self.dev = dev_set
        self.test = test_set
        self.sets = [train_set, dev_set, test_set]
        self.numcep = numcep
        self.numcontext = numcontext
        self.tower_feeder_count = max(len(get_available_gpus()), 1) if tower_feeder_count < 0 else tower_feeder_count
        self.threads_per_queue = threads_per_queue
        self.queue_capacity = queue_capacity

        self.ph_header = tf.placeholder(tf.int32, [])
        self.ph_x = tf.placeholder(tf.float32, [None, numcep + (2 * numcep * numcontext)])
        self.ph_x_length = tf.placeholder(tf.int32, [])
        self.ph_y = tf.placeholder(tf.int32, [None,])
        self.ph_y_length = tf.placeholder(tf.int32, [])
        self.ph_queue_selector = tf.placeholder(tf.int32, name='Queue_Selector')

        self._tower_feeders = [_TowerFeeder(self, i) for i in range(self.tower_feeder_count)]

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
        The DataSet has to be one of those that got passed into the constructor or None for the dummy one.
        '''
        index = ([None] + self.sets).index(data_set)
        assert index >= 0
        feed_dict[self.ph_queue_selector] = index

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
    def __init__(self, csvs, skip=0, limit=0, ascending=True, next_index=lambda i: i + 1):
        self.next_index = next_index
        self.files = None
        for csv in csvs:
            file = pandas.read_csv(csv)
            if self.files is None:
                self.files = file
            else:
                self.files = self.files.append(file)
        self.files = self.files.sort_values(by="wav_filesize", ascending=ascending) \
                         .ix[:, ["wav_filename", "transcript"]] \
                         .values[skip:]
        if limit > 0:
            self.files = self.files[:limit]

class _DataSetLoader(object):
    '''
    Internal class that represents an input queue with data from one of the DataSet objects.
    Each tower feeder will create and combine three data set loaders to one switchable queue.
    Keeps a ModelFeeder reference for accessing shared settings and placeholders.
    Keeps a DataSet reference to access its samples.
    If no data_set is provided (None), the loader will just enqueue dummy samples.
    '''
    def __init__(self, model_feeder, data_set):
        self._model_feeder = model_feeder
        self._data_set = data_set

        self._width = model_feeder.numcep + (2 * model_feeder.numcep * model_feeder.numcontext)
        self._dummy_sample = self._create_sample([self._width * [0]], 1, [0], 1)
        self._queue_lock = Lock()

        self.batch_size_queue = tf.FIFOQueue(1, tf.int32)
        self._batch_size_enqueue_op = self.batch_size_queue.enqueue([model_feeder.ph_header])
        self._batch_size_close_op = self.batch_size_queue.close(cancel_pending_enqueues=True)

        self.sample_queue = tf.PaddingFIFOQueue(shapes=[[None, self._width], [], [None,], []],
                                                dtypes=[tf.float32, tf.int32, tf.int32, tf.int32],
                                                capacity=model_feeder.queue_capacity if data_set else 1)
        self._sample_enqueue_op = self.sample_queue.enqueue([model_feeder.ph_x, model_feeder.ph_x_length, model_feeder.ph_y, model_feeder.ph_y_length])
        self._sample_close_op = self.sample_queue.close(cancel_pending_enqueues=True)

    def start_queue_threads(self, session, coord):
        '''
        Starts concurrent queue threads for reading samples from the data set.
        '''
        queue_threads = [Thread(target=self._populate_batch_queue, args=(session, coord))
                         for i in range(self._model_feeder.threads_per_queue if self._data_set else 1)]
        for queue_thread in queue_threads:
            coord.register_thread(queue_thread)
            queue_thread.daemon = True
            queue_thread.start()
        return queue_threads

    def close_queues(self, session):
        '''
        Closes the data set loader queues.
        '''
        session.run(self._batch_size_close_op)
        session.run(self._sample_close_op)

    def _create_sample(self, x, x_len, y, y_len):
        return { self._model_feeder.ph_x: x,
                 self._model_feeder.ph_x_length: x_len,
                 self._model_feeder.ph_y: y,
                 self._model_feeder.ph_y_length: y_len }

    def _enqueue_samples(self, session, coord, samples, batch_size=-1):
        try:
            with self._queue_lock:
                if not coord.should_stop():
                    session.run(self._batch_size_enqueue_op, feed_dict={
                        self._model_feeder.ph_header: len(samples) if batch_size < 0 else batch_size })
                for sample in samples:
                    if not coord.should_stop():
                        session.run(self._sample_enqueue_op, feed_dict=sample)
        except tf.errors.CancelledError:
            return

    def _populate_batch_queue(self, session, coord):
        '''
        Queue thread routine.
        '''
        # if we are the dummy queue, we'll just enqueue dummy samples
        if not self._data_set:
            while not coord.should_stop():
                self._enqueue_samples(session, coord, [self._dummy_sample], batch_size=0)
            return

        # we enqueue exactly file_count samples per epoch
        file_count = len(self._data_set.files)
        # current sample index (<0: wait, >=file_count: enqueue dummy samples)
        index = -1
        # keeping track of the maximum sample length within current batch
        max_len = 0
        # collecting samples of the current batch
        # each sample is represented by its enqueue-op's feed_dict
        samples = []
        while not coord.should_stop():
            # get next index from coordinator
            index = self._data_set.next_index(index)
            if index < 0:
                # the epoch has not started yet - wait and repeat
                time.sleep(1)
            elif index >= file_count:
                # there are no samples left
                if len(samples) > 0:
                    # let's just enqueue the remaining batch
                    self._enqueue_samples(session, coord, samples)
                    samples = []
                    max_len = 0
                else:
                    # sending trailing dummy samples to prevent remaining empty towers from blocking current job
                    self._enqueue_samples(session, coord, [self._dummy_sample], batch_size=0)
            else:
                # we got a sample to enqueue
                wav_file, transcript = self._data_set.files[index]
                # preparing the audio
                source = audiofile_to_input_vector(wav_file, self._model_feeder.numcep, self._model_feeder.numcontext)
                source_len = len(source)
                # let's fail, if the sample alone already exceeds the batch len threshold
                assert source_len <= self._model_feeder.len_threshold
                # preparing the text
                target = text_to_char_array(transcript)
                target_len = len(target)
                # create the sample's feed_dict
                sample = self._create_sample(source, source_len, target, target_len)
                # compute overall length of the hypothetical batch (with current sample contained)
                # -> in PaddingFIFOQueue's memory model all item slots are padded to the length of the biggest item
                batch_len = (len(samples) + 1) * max(max_len, source_len)
                # if the hypothetical batch's length exceeds our length threshold...
                if batch_len > self._model_feeder.len_threshold:
                    # we first enqueue the batch without the current sample...
                    print ('enqueuing %d samples' % len(samples))
                    self._enqueue_samples(session, coord, samples)
                    # and begin a new batch with the current sample
                    samples = [sample]
                    max_len = source_len
                else:
                    # if not, we just add the current sample to the batch
                    # this new batch should still fit into memory
                    samples.append(sample)
                    max_len = max(max_len, source_len)

class _TowerFeeder(object):
    '''
    Internal class that represents switchable batch_size and sample queues for one tower.
    It creates, owns and combines four _DataSetLoader instances (train, dev, test, dummy).
    Keeps a ModelFeeder reference for accessing shared settings and placeholders.
    '''
    def __init__(self, model_feeder, index):
        self._model_feeder = model_feeder
        self.index = index
        self._loaders = [_DataSetLoader(model_feeder, None)] + \
                        [_DataSetLoader(model_feeder, data_set) for data_set in model_feeder.sets]
        self._batch_size_queues = [loader.batch_size_queue for loader in self._loaders]
        self._sample_queues =     [loader.sample_queue for loader in self._loaders]
        self._batch_size_queue = tf.QueueBase.from_list(model_feeder.ph_queue_selector, self._batch_size_queues)
        self._sample_queue =     tf.QueueBase.from_list(model_feeder.ph_queue_selector, self._sample_queues)

    def next_batch(self):
        '''
        Draw the next batch from the combined switchable queue.
        '''
        batch_size = self._batch_size_queue.dequeue()
        batch_size = tf.Print(batch_size, [batch_size], 'Dequeueing batch size: ')
        # to dequeue the dummy sample, dequeue_size has to be 1 in case of batch_size=0
        dequeue_size = tf.maximum(batch_size, 1)
        source, source_lengths, target, target_lengths = self._sample_queue.dequeue_many(dequeue_size)
        sparse_labels = ctc_label_dense_to_sparse(target, target_lengths, batch_size)
        return batch_size, source, source_lengths, sparse_labels

    def start_queue_threads(self, session, coord):
        '''
        Starts the queue threads of all owned _DataSetLoader instances.
        '''
        queue_threads = []
        for loader in self._loaders:
            queue_threads += loader.start_queue_threads(session, coord)
        return queue_threads

    def close_queues(self, session):
        '''
        Closes queues of all owned _DataSetLoader instances.
        '''
        for loader in self._loaders:
            loader.close_queues(session)

