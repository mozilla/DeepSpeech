import pandas
import tensorflow as tf
import time
import Queue

from threading import Thread, Lock
from math import ceil
from six.moves import range
from util.audio import audiofile_to_input_vector
from util.text import ctc_label_dense_to_sparse, text_to_char_array
from util.log import Logger

log = Logger(id='feeding', caption='Feeding')

class DataSet(object):
    '''
    Represents a collection of audio samples and their respective transcriptions.
    Takes a set of CSV files produced by importers in "bin/".
    '''
    def __init__(self, csvs, batch_size=0, skip=0, limit=0, ascending=True):
        '''
        Constructs a DataSet object from a list of provided csv files.
        'csvs' - List of filenames of comma separated value files that have the columns
            - wav_filename - Filename of the voice audio file
            - wav_filesize - The file size of the voice audio file
            - transcript - The transription of the voice sample
        'batch_size' - The batch size for this data set - if 0 (default),
            it will be computed dynamically for each applied batch
        'skip' - how many samples to skip from the beginning of the DataSet's sample sequence
        'limit' - limits the the number of samples to load from the DataSet
        'ascending' - if the samples of the data set should be ordered ascending (True)
            of descending (False) by their file sizes
        '''
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
        self.batch_size = batch_size

class ModelFeeder(object):
    '''
    Feeds data into a model.
    Feeding is parallelized by independent units called tower feeders (usually one per GPU).
    Each tower feeder provides data from three runtime switchable sources (train, dev, test).
    These sources are to be provided by three DataSet instances who's references are kept.
    Creates, owns and delegates to num_tower_feeders internal tower feeder objects.
    '''
    def __init__(self,
                 towers_per_worker,
                 worker_index,
                 num_cep,
                 num_context,
                 alphabet,
                 num_sample_loaders_per_set,
                 num_samples_loader_buffer,
                 min_tower_memory,
                 queue_capacity,
                 allocate_indices=lambda n: 0):

        self.towers_per_worker = towers_per_worker
        self.worker_index = worker_index
        self.num_cep = num_cep
        self.num_context = num_context
        self.alphabet = alphabet
        self.num_sample_loaders_per_set = num_sample_loaders_per_set
        self.num_samples_loader_buffer = num_samples_loader_buffer
        self.min_tower_memory = min_tower_memory
        self.queue_capacity = queue_capacity
        self.allocate_indices = allocate_indices

        self.ph_x = tf.placeholder(tf.float32, [None, num_cep + (2 * num_cep * num_context)])
        self.ph_x_length = tf.placeholder(tf.int32, [])
        self.ph_y = tf.placeholder(tf.int32, [None,])
        self.ph_y_length = tf.placeholder(tf.int32, [])

        self.width = num_cep + (2 * num_cep * num_context)
        self.dummy_sample = self._create_sample([self.width * [0]], 1, [0], 1)

        # protects the feeder state
        self._lock = Lock()
        # special dummy batch
        self._dummy = (0, [None])
        # queue of batches that are ready to be enqueued on towers
        # a batch is a tuple of the batch size and a list of sample tuples
        self.queue = Queue.Queue(maxsize=len(self.towers_per_worker[self.worker_index]))
        # take care data set member exists
        self._data_set = None
        # set it again to initialize all structures for feeding
        self.start_data_set(None)

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

        class WorkerInfo: pass
        class TowerInfo: pass
        self.towers = []
        self.workers = []
        for worker_index, towers in enumerate(towers_per_worker):
            worker = WorkerInfo()
            worker.index = worker_index
            worker.towers = []
            for tower_index, tower_memory in enumerate(towers_per_worker[worker_index]):
                tower = TowerInfo()
                tower.worker = worker
                tower.index = tower_index
                tower.memory = tower_memory
                tower.queue, tower.enqueue, tower.close = self._create_queue(worker_index, tower_index)
                worker.towers.append(tower)
                self.towers.append(tower)
            self.workers.append(worker)

        # local tower feeders - each one typically feeding into a GPU
        self._tower_feeders = [_TowerFeeder(self, tower.index, tower.queue, tower.enqueue, tower.close) for tower in self.workers[self.worker_index].towers]

    def next_batch(self, worker_index, tower_index):
        queue = self.workers[worker_index].towers[tower_index].queue
        _, batch_size, _, _ = queue.dequeue(name='Batch_Size_Dequeue')
        dequeue_size = tf.maximum(batch_size, 1)
        source, source_lengths, target, target_lengths = queue.dequeue_many(dequeue_size, name='Samples_Dequeue')
        sparse_labels = ctc_label_dense_to_sparse(target, target_lengths, batch_size)
        return batch_size, source, source_lengths, sparse_labels

    def start_queue_threads(self, session, coord):
        '''
        Starts required queue threads on all loaders and tower feeders.
        '''
        threads = [Thread(target=self._main_loop, args=(coord,))] + \
                  [Thread(target=self._load_sample, args=(coord,))
                   for i in range(self.num_sample_loaders_per_set)]
        for thread in threads:
            coord.register_thread(thread)
            thread.daemon = True
            thread.start()
        for tower_feeder in self._tower_feeders:
            threads.append(tower_feeder.start_queue_thread(session, coord))
        log.debug('Started queue threads')
        return threads

    def get_state_report(self):
        return 'To load: %d, Loading: %d, Loaded: %d, Enqueued: %d' % \
            (len(self._to_load), len(self._loading), len(self._loaded), self._enqueued)

    def close_queues(self, session):
        '''
        Closes queues of all tower feeders.
        '''
        for tower_feeder in self._tower_feeders:
            tower_feeder.close_queues(session)
        log.debug('Closed queues')

    def get_data_set(self):
        '''
        Retrives current DataSet.
        '''
        return self._data_set

    def start_data_set(self, data_set):
        '''
        Switches to a different source DataSet.
        '''
        with self._lock:
            # counter for debugging
            self._enqueued = 0
            # current data set to feed into queues
            self._data_set = data_set
            # list of indices of samples that are to be loaded
            self._to_load = []
            # list of indices of samples that are currently loading
            self._loading = []
            # list of sample tuples (source, source's length, target and target's length) that got loaded
            self._loaded = []
            # draining the queue (index=-2 hinders dummy loop from flooding queue)
            self._index = -2
            while not self.queue.empty():
                self.queue.get_nowait()
            # first index of last retrieved index allocation (<0: wait, >=file_count: enqueue dummy batches)
            self._index = -1
        log.debug('Started data set')

    def _create_sample(self, x, x_len, y, y_len):
        return { self.ph_x: x, self.ph_x_length: x_len, self.ph_y: y, self.ph_y_length: y_len }

    def _create_queue(self, worker_index, tower_index):
        name = 'PFQ-%d-%s' % (worker_index, tower_index)
        queue = tf.PaddingFIFOQueue(shapes=[[None, self.width], [], [None,], []],
                                    dtypes=[tf.float32, tf.int32, tf.int32, tf.int32],
                                    capacity=self.queue_capacity,
                                    name=name,
                                    shared_name=name)
        enqueue = queue.enqueue([self.ph_x, self.ph_x_length, self.ph_y, self.ph_y_length])
        close = queue.close(cancel_pending_enqueues=True)
        return (queue, enqueue, close)

    def _main_loop(self, coord):
        # if we are the dummy queue, we'll just enqueue dummy batches
        while not coord.should_stop():
            if not self._data_set:
                # only enqueue, if we're not in a draining process
                if self._index != -2:
                    try:
                        # trying for one second to enqueue the dummy
                        self.queue.put(self._dummy, True, 1)
                        log.step('Enqueued dummy sample')
                    except Queue.Full:
                        # just ignore the timeout and loop on
                        pass
                continue
            # the overall number of samples to enqueue
            file_count = len(self._data_set.files)
            # sample batch that will be queued
            batch = None
            with self._lock:
                # number of indices to refill
                number = self.num_samples_loader_buffer - len(self._to_load)
                # if we have to allocate more than 50% of the fill-rate,
                # we query for more indices...
                # Note: This results in allocations of big blocks of consecutive samples
                if number > self.num_samples_loader_buffer / 2 and self._index < file_count:
                    # let the coordinator allocate a 'number' of indices for us
                    self._index = self.allocate_indices(number)
                    # unpacking the index allocation...
                    for j in range(number):
                        if self._index >= 0 and self._index < file_count:
                            # handing sample index over to sample loader threads
                            self._to_load.append(self._index)
                        else:
                            break
                        self._index += 1
                    # Note: the case index=-1 should've survived unpacking
                # if we got samples loaded by sample loaders, we could construct a batch...
                if len(self._loaded) > 0:
                    if self._data_set.batch_size > 0 and self._data_set.batch_size <= len(self._loaded):
                        # Static Batch Size
                        # -----------------
                        # cut the batch from loaded samples
                        batch = self._loaded[:self._data_set.batch_size]
                        # and keep the rest
                        self._loaded = self._loaded[self._data_set.batch_size:]
                    else:
                        # Dynamic Batch Size
                        # ------------------
                        # ordering samples by audio length (source_len) to achieve good packing of the
                        # batch to maximize memory utilization of PaddingFIFOQueue
                        # Not*e: Our thresholding is based and calibrated on audio data size, as this
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
                            if (i + 1) * self._loaded[i][1] > self.len_threshold:
                                # if so, we build a batch with all samples till the (i-1)-th one
                                # (as this already passed our test during last iteration)...
                                assert i > 0 # fail, if first sample alone (i = 0) is already too big
                                # cut the batch (samples 0 to i-1) from loaded samples
                                batch = self._loaded[:i]
                                # and keep the rest
                                self._loaded = self._loaded[i:]
                                break
                    # if there is no batch yet and there are no samples under way,
                    if not batch and (len(self._to_load) + len(self._loading) == 0 or len(self._loaded) >= self.num_samples_loader_buffer):
                        # we build a (smaller) remainder batch from all loaded samples
                        # Note: This is supposedly the last batch of the current epoch on this node.
                        batch = self._loaded
                        self._loaded = []
            if batch:
                # enqueuing the batch in a non-blocking way to allow coordinator induced exiting
                while not coord.should_stop():
                    try:
                        # trying for one second to enqueue the batch
                        self.queue.put((len(batch), batch), True, 1)
                        log.step('Enqueued batch with %d samples' % len(batch))
                        break
                    except Queue.Full:
                        # we ran into timeout - repeat...
                        pass
                # keeping track of enqueued samples
                self._enqueued += len(batch)
            else:
                if self._index >= file_count:
                    # sending trailing dummy batches to prevent remaining empty towers from blocking current job
                    try:
                        # trying for one second to enqueue the batch
                        self.queue.put(self._dummy, True, 1)
                        log.step('Enqueued trailing dummy sample')
                    except Queue.Full:
                        # we ran into timeout - restart at the very top (forgetting this dummy sample)...
                        pass
                else:
                    # the epoch hasn't started yet (index < 0) or
                    # not enough samples got loaded yet -> wait a sec and repeat
                    time.sleep(1)

    def _load_sample(self, coord):
        while not coord.should_stop():
            # reserve next sample index to load
            index = -1
            with self._lock:
                if len(self._to_load) > 0 and len(self._loaded) < self.num_samples_loader_buffer:
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
            source = audiofile_to_input_vector(wav_file, self.num_cep, self.num_context)
            source_len = len(source)
            # let's fail, if the sample alone already exceeds the batch lenght threshold
            assert source_len <= self.len_threshold
            # preparing the text
            target = text_to_char_array(transcript, self.alphabet)
            target_len = len(target)
            with self._lock:
                # handing loaded sample over to the main thread,
                # if there was no reset in the meantime
                if index in self._loading:
                    self._loading.remove(index)
                    self._loaded.append((source, source_len, target, target_len))

class _TowerFeeder(object):
    '''
    Internal class that represents an input queue with data from one of the DataSet objects.
    Each tower feeder will create and combine three batch feeders to one switchable queue.
    Keeps a ModelFeeder reference for accessing shared settings and placeholders.
    Keeps a DataSet reference to access its samples.
    If no data_set is provided (None), the loader will just enqueue dummy samples.
    '''
    def __init__(self, model_feeder, tower_index, queue, enqueue, close):
        self.model_feeder = model_feeder
        self.tower_index = tower_index
        self._queue = queue
        self._enqueue = enqueue
        self._close = close
        self._size = self._queue.size()

    def start_queue_thread(self, session, coord):
        '''
        Starts the queue thread for getting batches from the data set
        and enqueuing them on sample_queue.
        '''
        queue_thread = Thread(target=self._populate_sample_queue, args=(session, coord))
        coord.register_thread(queue_thread)
        queue_thread.daemon = True
        queue_thread.start()
        return queue_thread

    def close_queues(self, session):
        '''
        Closes the data set loader queues.
        '''
        session.run(self._close)

    def _populate_sample_queue(self, session, coord):
        '''
        Queue thread routine.
        '''
        log.debug('Tower %d enqueueing started' % self.tower_index)
        # all of the following dequeue/enqueue operations are non-blocking
        # to ensure graceful ending of the thread (querying coord.should_stop)
        while not coord.should_stop():
            # polling loop - dequeueing/waiting for the next batch
            while not coord.should_stop():
                try:
                    # getting/waiting for the next batch
                    batch_size, samples = self.model_feeder.queue.get(True, 1)
                    # we got a batch - exit polling loop
                    break
                except Queue.Empty:
                    # we ran into timeout getting the next batch - poll on
                    pass
            try:
                # to prevent flooding, dummy samples are only enqueued onto an empty queue
                while not coord.should_stop() and batch_size == 0 and session.run(self._size) > 0:
                    time.sleep(1)
                # create header sample
                feed_dict = self.model_feeder._create_sample([self.model_feeder.width * [0]], batch_size, [0], 0)
                if not coord.should_stop():
                    # enqueue it
                    session.run(self._enqueue, feed_dict=feed_dict)
                # enqueue all regular samples of the batch
                for sample in samples:
                    # prepare sample feed_dict
                    feed_dict = self.model_feeder._create_sample(*sample) if batch_size > 0 else \
                                self.model_feeder.dummy_sample
                    if not coord.should_stop():
                        # enqueue it
                        session.run(self._enqueue, feed_dict=feed_dict)
            except (tf.errors.CancelledError, tf.errors.OutOfRangeError):
                # this is expected under certain circumstances - just exiting
                log.debug('Tower %d enqueueing ended by closed queue' % self.tower_index)
                return
        # regular end
        log.debug('Tower %d enqueueing ended by coordinator' % self.tower_index)
