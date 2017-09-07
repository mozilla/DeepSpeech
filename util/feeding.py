import pandas
import tensorflow as tf
import time
import Queue

from threading import Thread, Lock
from math import ceil
from six.moves import range
from util.audio import audiofile_to_input_vector
from util.text import ctc_label_dense_to_sparse, text_to_char_array

class ModelFeeder(object):
    '''
    Feeds data into a model.
    Feeding is parallelized by independent units called tower feeders (usually one per GPU).
    Each tower feeder provides data from three runtime switchable sources (train, dev, test).
    These sources are to be provided by three DataSet instances who's references are kept.
    Creates, owns and delegates to num_tower_feeders internal tower feeder objects.
    '''
    def __init__(self,
                 train_set,
                 dev_set,
                 test_set,
                 num_cep,
                 num_context,
                 num_tower_feeders,
                 num_sample_loaders_per_set,
                 num_samples_loader_buffer,
                 min_tower_memory,
                 queue_capacity):

        self.train = train_set
        self.dev = dev_set
        self.test = test_set
        self.sets = [train_set, dev_set, test_set]
        self.num_cep = num_cep
        self.num_context = num_context
        self.num_tower_feeders = num_tower_feeders
        self.num_sample_loaders_per_set = num_sample_loaders_per_set
        self.num_samples_loader_buffer = num_samples_loader_buffer
        self.min_tower_memory = min_tower_memory
        self.queue_capacity = queue_capacity

        self.ph_header = tf.placeholder(tf.int32, [])
        self.ph_x = tf.placeholder(tf.float32, [None, num_cep + (2 * num_cep * num_context)])
        self.ph_x_length = tf.placeholder(tf.int32, [])
        self.ph_y = tf.placeholder(tf.int32, [None,])
        self.ph_y_length = tf.placeholder(tf.int32, [])
        self.ph_queue_selector = tf.placeholder(tf.int32, [], name='Queue_Selector')

        # Bisected threshold t1 at memory m1
        m1 = 3816882176
        t1 = 2000

        # Bisected threshold t2 at memory m2
        m2 = 7994143540
        t2 = 11000

        # threshold by a linear function derived from the two "measurements"
        self.len_threshold = (min_tower_memory * (t2-t1) + m2*t1 - m1*t2) // (m2-m1)

        # Threshold measurement procedure
        # ===============================
        # This should be done in case of bigger graph changes and/or OOM failures:
        # 1 - Uncomment the following prints to easily see the current memory/threshold config
        # print('#' * 100)
        # print('Memory: %d' % min_tower_memory)
        # print('Threshold: %d' % self.len_threshold)
        # print('#' * 100)
        # 2 - Do a single GPU run (-> CUDA_VISIBLE_DEVICES) for at least 5 batches
        # 3 - Depending on the outcome, either lower (in case of an OOM) or increase the threshold
        #     and assign it to self.len_threshold by copy/commenting the above line
        #     "self.len_threshold = ..."
        # 4 - Continue with step 2/3, till you bisected the new threshold to the maximum value that
        #     runs without an OOM failure (don't go beyond a precision/buffer of 100 units) and assign
        #     the final Memory/Threshold values to m2/t2 variables above.
        # 5 - Allocate 4 GB of the GPU memory for the next run by using the --gpu_allocation parameter
        #     or (if this doesn't work) by using a tool like the one posted here to block remaining memory:
        #     https://devtalk.nvidia.com/default/topic/726765/need-a-little-tool-to-adjust-the-vram-size/
        # 6 - Repeat steps 2-4 to get values m1/t1
        # 7 - Reactivate the original line "self.len_threshold = ..."
        # 8 - Do some further tests (with more batches and memory allocations between m1 and m2)
        #     and decrease t1 and t2 if there are still OOMs
        # 9 - If everything works, don't forget to comment the prints again

        self._sets = [None, train_set, dev_set, test_set]
        self.loaders = [_BatchLoader(self, s) for s in self._sets]
        self._tower_feeders = [_TowerFeeder(self, i) for i in range(self.num_tower_feeders)]

    def start_queue_threads(self, session, coord):
        '''
        Starts required queue threads on all loaders and tower feeders.
        '''
        queue_threads = []
        for loader in self.loaders:
            queue_threads += loader.start_threads(coord)
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
        index = self._sets.index(data_set)
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
    def __init__(self, csvs, skip=0, limit=0, ascending=True, get_indices=lambda n: 0):
        self.get_indices = get_indices
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

class _BatchLoader(object):
    '''
    Internal class that loads batches from a data set.
    _BatchFeeder instances will distribute them to all towers and their respective input queues.
    Keeps a ModelFeeder reference for accessing shared settings and placeholders.
    Keeps a DataSet reference to access its samples.
    If no data_set is provided (None), the loader will just enqueue dummy samples.
    '''
    def __init__(self, model_feeder, data_set):
        self._model_feeder = model_feeder
        self._data_set = data_set
        # protects the lists from undefined state
        self._lock = Lock()
        # list of indices of samples that are to be loaded
        self._to_load = []
        # list of indices of samples that are currently loading
        self._loading = []
        # list of sample tuples (source, source's length, target and target's length) that got loaded
        self._loaded = []
        # queue of batches that are ready to be enqueued on towers
        # a batch is a tuple of the batch size and a list of sample tuples
        self.queue = Queue.Queue(maxsize=2 * self._model_feeder.num_tower_feeders)
        # special dummy batch
        self._dummy = (0, [None])

    def start_threads(self, coord):
        '''
        Starts concurrent threads for reading samples from the data set.
        '''
        threads = [Thread(target=self._main_loop, args=(coord,))] + \
                  [Thread(target=self._load_sample, args=(coord,))
                   for i in range(self._model_feeder.num_sample_loaders_per_set if self._data_set else 0)]
        for thread in threads:
            coord.register_thread(thread)
            thread.daemon = True
            thread.start()
        return threads

    def _main_loop(self, coord):
        # if we are the dummy queue, we'll just enqueue dummy batches
        if not self._data_set:
            while not coord.should_stop():
                self.queue.put(self._dummy)
            return

        # first index of last retrieved index allocation (<0: wait, >=file_count: enqueue dummy batches)
        index = -1
        # we enqueue exactly file_count samples per epoch
        file_count = len(self._data_set.files)
        while not coord.should_stop():
            # sample batch that will be queued
            batch = None
            with self._lock:
                # number of indices to refill
                number = self._model_feeder.num_samples_loader_buffer - len(self._to_load)
                # if we have to allocate more than 50% of the fill-rate,
                # we query for more indices...
                # Note: This results in allocations of big blocks of consecutive samples
                if number > self._model_feeder.num_samples_loader_buffer / 2:
                    # let the coordinator allocate a 'number' of indices for us
                    index = self._data_set.get_indices(number)
                    # unpacking the index allocation...
                    for j in range(number):
                        if index >= 0 and index < file_count:
                            # handing sample index over to sample loader threads
                            self._to_load.append(index)
                        else:
                            break
                        index = index + 1
                    # Note: the case index=-1 should've survived unpacking
                # if we got samples loaded by sample loaders, we could construct a batch...
                if len(self._loaded) > 0:
                    # ordering samples by audio length (source_len) to achieve good packing of the
                    # batch to maximize memory utilization of PaddingFIFOQueue
                    # Note: Our thresholding is based and calibrated on audio data size, as this
                    # is supposedly by far the most memory critical factor.
                    self._loaded = sorted(self._loaded, key=lambda sample: sample[1])
                    # find the biggest batch from all loaded samples...
                    for i in range(len(self._loaded)):
                        # test, if current hypothetical batch already exceeds the length threshold...
                        # Note: PaddingFIFOQueue is padding all samples to the size of the biggest
                        # one (the current i-th) and requires (i + 1) slots.
                        # Note: The for-loop will only produce a batch, if a sample results in a batch
                        # that would exceed the threshold. This ensures that we are not enqueuing small
                        # batches just because there are not enough samples loaded yet. The code directly
                        # following the loop will care about the other/remainder case.
                        if (i + 1) * self._loaded[i][1] > self._model_feeder.len_threshold:
                            # if so, we build a batch with all samples till the (i-1)-th one
                            # (as this already passed our test during last iteration)...
                            assert i > 0 # fail, if first sample alone (i = 0) is already too big
                            # cut the batch (samples 0 to i-1) from loaded samples
                            batch = self._loaded[:i]
                            # and keep the rest
                            self._loaded = self._loaded[i:]
                            break
                    # if there is no batch yet and there are no samples under way,
                    if not batch and len(self._to_load) + len(self._loading) == 0:
                        # we build a (smaller) remainder batch from all loaded samples
                        # Note: This is supposedly the last batch of the current epoch on this node.
                        batch = self._loaded
                        self._loaded = []
            if batch:
                # enqueuing the batch
                self.queue.put((len(batch), batch))
            else:
                if index >= file_count:
                    # sending trailing dummy batches to prevent remaining empty towers from blocking current job
                    self.queue.put(self._dummy)
                else:
                    # the epoch hasn't started yet (index < 0) or
                    # not enough samples got loaded yet (but len(self._to_load) + len(self._loading) > 0)
                    # -> wait a sec and repeat
                    time.sleep(1)

    def _load_sample(self, coord):
        while not coord.should_stop():
            # reserve next sample index to load
            index = -1
            with self._lock:
                if len(self._to_load) > 0:
                    index = self._to_load.pop(0)
                    # main thread needs to know, how many samples are under way
                    self._loading.append(index)
            # it's better to sleep outside the lock...
            if index < 0:
                # wait for the main thread to provide some sample indices
                time.sleep(1)
                continue
            # we got a sample to process
            wav_file, transcript = self._data_set.files[index]
            # preparing the audio
            source = audiofile_to_input_vector(wav_file, self._model_feeder.num_cep, self._model_feeder.num_context)
            source_len = len(source)
            # let's fail, if the sample alone already exceeds the batch lenght threshold
            assert source_len <= self._model_feeder.len_threshold
            # preparing the text
            target = text_to_char_array(transcript)
            target_len = len(target)
            with self._lock:
                # handing loaded sample over to the main thread
                self._loading.remove(index)
                self._loaded.append((source, source_len, target, target_len))

class _BatchFeeder(object):
    '''
    Internal class that represents an input queue with data from one of the DataSet objects.
    Each tower feeder will create and combine three batch feeders to one switchable queue.
    Keeps a ModelFeeder reference for accessing shared settings and placeholders.
    Keeps a DataSet reference to access its samples.
    If no data_set is provided (None), the loader will just enqueue dummy samples.
    '''
    def __init__(self, model_feeder, batch_loader):
        self._model_feeder = model_feeder
        self._batch_loader = batch_loader

        self._width = model_feeder.num_cep + (2 * model_feeder.num_cep * model_feeder.num_context)
        self._dummy_sample = self._create_sample([self._width * [0]], 1, [0], 1)

        self.sample_queue = tf.PaddingFIFOQueue(shapes=[[None, self._width], [], [None,], []],
                                                dtypes=[tf.float32, tf.int32, tf.int32, tf.int32],
                                                capacity=model_feeder.queue_capacity)
        self._sample_queue_size = self.sample_queue.size()
        self._sample_enqueue_op = self.sample_queue.enqueue([model_feeder.ph_x, model_feeder.ph_x_length, model_feeder.ph_y, model_feeder.ph_y_length])
        self._sample_close_op = self.sample_queue.close(cancel_pending_enqueues=True)

    def start_queue_thread(self, session, coord):
        '''
        Starts the queue thread for getting batches from the data set
        and enqueuing them on sample_queue.
        '''
        queue_thread = Thread(target=self._populate_batch_queues, args=(session, coord))
        coord.register_thread(queue_thread)
        queue_thread.daemon = True
        queue_thread.start()
        return queue_thread

    def close_queues(self, session):
        '''
        Closes the data set loader queues.
        '''
        session.run(self._sample_close_op)

    def _create_sample(self, x, x_len, y, y_len):
        return { self._model_feeder.ph_x: x,
                 self._model_feeder.ph_x_length: x_len,
                 self._model_feeder.ph_y: y,
                 self._model_feeder.ph_y_length: y_len }

    def _populate_batch_queues(self, session, coord):
        '''
        Queue thread routine.
        '''
        while not coord.should_stop():
            batch_size, samples = self._batch_loader.queue.get()
            if batch_size == 0 and session.run(self._sample_queue_size) > 0:
                continue
            try:
                if not coord.should_stop():
                    feed_dict = self._create_sample([self._width * [0]], batch_size, [0], 0)
                    session.run(self._sample_enqueue_op, feed_dict=feed_dict)
                for sample in samples:s
                    if not coord.should_stop():
                        feed_dict = self._create_sample(*sample) if batch_size > 0 else self._dummy_sample
                        session.run(self._sample_enqueue_op, feed_dict=feed_dict)
            except tf.errors.CancelledError:
                return

class _TowerFeeder(object):
    '''
    Internal class that represents switchable batch_size and sample queues for one tower.
    It creates, owns and combines four _BatchFeeder instances (train, dev, test, dummy).
    Keeps a ModelFeeder reference for accessing batch loaders, shared settings and placeholders.
    '''
    def __init__(self, model_feeder, index):
        self._model_feeder = model_feeder
        self.index = index
        self._batch_feeders = [_BatchFeeder(model_feeder, loader) for loader in model_feeder.loaders]
        self._sample_queues = [feeder.sample_queue for feeder in self._batch_feeders]
        self._sample_queue = tf.QueueBase.from_list(model_feeder.ph_queue_selector, self._sample_queues)

    def next_batch(self):
        '''
        Draw the next batch from the combined switchable queue.
        '''
        _, batch_size, _, _ = self._sample_queue.dequeue(name='Batch_Size_Dequeue')
        # to dequeue the dummy sample, dequeue_size has to be 1 in case of batch_size=0
        dequeue_size = tf.maximum(batch_size, 1)
        source, source_lengths, target, target_lengths = self._sample_queue.dequeue_many(dequeue_size, name='Samples_Dequeue')
        sparse_labels = ctc_label_dense_to_sparse(target, target_lengths, batch_size)
        return batch_size, source, source_lengths, sparse_labels

    def start_queue_threads(self, session, coord):
        '''
        Starts the queue threads of all owned _BatchFeeder instances.
        '''
        queue_threads = []
        for feeder in self._batch_feeders:
            queue_threads.append(feeder.start_queue_thread(session, coord))
        return queue_threads

    def close_queues(self, session):
        '''
        Closes queues of all owned _BatchFeeder instances.
        '''
        for feeder in self._batch_feeders:
            feeder.close_queues(session)

