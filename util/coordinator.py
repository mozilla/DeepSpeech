from __future__ import absolute_import, division, print_function

import pickle
import tensorflow as tf

from datetime import datetime
from six.moves import zip, range, filter, urllib, BaseHTTPServer
from threading import Thread, Lock
from util.config import Config
from util.flags import FLAGS
from util.logging import log_info, log_error, log_debug, log_warn, log_traffic


# Execution
# =========

# For reporting we also need a standard way to do time measurements.
def stopwatch(start_duration=0):
    r'''
    This function will toggle a stopwatch.
    The first call starts it, second call stops it, third call continues it etc.
    So if you want to measure the accumulated time spent in a certain area of the code,
    you can surround that code by stopwatch-calls like this:

    .. code:: python

        fun_time = 0 # initializes a stopwatch
        [...]
        for i in range(10):
          [...]
          # Starts/continues the stopwatch - fun_time is now a point in time (again)
          fun_time = stopwatch(fun_time)
          fun()
          # Pauses the stopwatch - fun_time is now a duration
          fun_time = stopwatch(fun_time)
        [...]
        # The following line only makes sense after an even call of :code:`fun_time = stopwatch(fun_time)`.
        print 'Time spent in fun():', format_duration(fun_time)

    '''
    if start_duration == 0:
        return datetime.utcnow()
    else:
        return datetime.utcnow() - start_duration

def format_duration(duration):
    '''Formats the result of an even stopwatch call as hours:minutes:seconds'''
    duration = duration if isinstance(duration, int) else duration.seconds
    m, s = divmod(duration, 60)
    h, m = divmod(m, 60)
    return '%d:%02d:%02d' % (h, m, s)


# String constants for different services of the web handler
PREFIX_NEXT_INDEX = '/next_index_'
PREFIX_GET_JOB = '/get_job_'

# Global ID counter for all objects requiring an ID
id_counter = 0

def new_id():
    '''Returns a new ID that is unique on process level. Not thread-safe.

    Returns:
        int. The new ID
    '''
    global id_counter
    id_counter += 1
    return id_counter

class WorkerJob(object):
    '''Represents a job that should be executed by a worker.

    Args:
        epoch_id (int): the ID of the 'parent' epoch
        index (int): the epoch index of the 'parent' epoch
        set_name (str): the name of the data-set - one of 'train', 'dev'
        steps (int): the number of `session.run` calls
    '''
    def __init__(self, epoch_id, index, set_name, steps):
        self.id = new_id()
        self.epoch_id = epoch_id
        self.index = index
        self.worker = -1
        self.set_name = set_name
        self.steps = steps
        self.loss = -1
        self.samples = []

    def __str__(self):
        return 'Job (ID: %d, worker: %d, epoch: %d, set_name: %s)' % (self.id, self.worker, self.index, self.set_name)

class Epoch(object):
    '''Represents an epoch that should be executed by the Training Coordinator.
    Creates `num_jobs` `WorkerJob` instances in state 'open'.

    Args:
        index (int): the epoch index of the 'parent' epoch
        num_jobs (int): the number of jobs in this epoch

    Kwargs:
        set_name (str): the name of the data-set - one of 'train', 'dev'
    '''
    def __init__(self, coord, index, num_jobs, set_name='train'):
        self.id = new_id()
        self.coord = coord
        self.index = index
        self.num_jobs = num_jobs
        self.set_name = set_name
        self.loss = -1
        self.jobs_open = []
        self.jobs_running = []
        self.jobs_done = []
        for i in range(self.num_jobs):
            self.jobs_open.append(WorkerJob(self.id, self.index, self.set_name, FLAGS.iters_per_worker))

    def name(self):
        '''Gets a printable name for this epoch.

        Returns:
            str. printable name for this epoch
        '''
        if self.index >= 0:
            ename = ' of Epoch %d' % self.index
        else:
            ename = ''
        if self.set_name == 'train':
            return 'Training%s' % ename
        else:
            return 'Validation%s' % ename

    def get_job(self, worker):
        '''Gets the next open job from this epoch. The job will be marked as 'running'.

        Args:
            worker (int): index of the worker that takes the job

        Returns:
            WorkerJob. job that has been marked as running for this worker
        '''
        if len(self.jobs_open) > 0:
            job = self.jobs_open.pop(0)
            self.jobs_running.append(job)
            job.worker = worker
            return job
        else:
            return None

    def finish_job(self, job):
        '''Finishes a running job. Removes it from the running jobs list and adds it to the done jobs list.

        Args:
            job (WorkerJob): the job to put into state 'done'
        '''
        index = next((i for i in range(len(self.jobs_running)) if self.jobs_running[i].id == job.id), -1)
        if index >= 0:
            self.jobs_running.pop(index)
            self.jobs_done.append(job)
            log_traffic('%s - Moved %s from running to done.' % (self.name(), job))
        else:
            log_warn('%s - There is no job with ID %d registered as running.' % (self.name(), job.id))

    def done(self):
        '''Checks, if all jobs of the epoch are in state 'done'.

        Returns:
            bool. if all jobs of the epoch are 'done'
        '''
        if len(self.jobs_open) == 0 and len(self.jobs_running) == 0:
            num_jobs = len(self.jobs_done)
            if num_jobs > 0:
                jobs = self.jobs_done
                self.jobs_done = []
                if not self.num_jobs == num_jobs:
                    log_warn('%s - Number of steps not equal to number of jobs done.' % (self.name()))

                agg_loss = 0.0

                for i in range(num_jobs):
                    job = jobs.pop(0)
                    agg_loss += job.loss

                self.loss = agg_loss / num_jobs

                # if the job was for validation dataset then append it to the COORD's _loss for early stop verification
                if (FLAGS.early_stop is True) and (self.set_name == 'dev'):
                    self.coord._dev_losses.append(self.loss)

            return True
        return False

    def job_status(self):
        '''Provides a printable overview of the states of the jobs of this epoch.

        Returns:
            str. printable overall job state
        '''
        return '%s - jobs open: %d, jobs running: %d, jobs done: %d' % (self.name(), len(self.jobs_open), len(self.jobs_running), len(self.jobs_done))

    def __str__(self):
        if not self.done():
            return self.job_status()

        return '%s - loss: %f' % (self.name(), self.loss)


class TrainingCoordinator(object):
    ''' Central training coordination class.
    Used for distributing jobs among workers of a cluster.
    Instantiated on all workers, calls of non-chief workers will transparently
    HTTP-forwarded to the chief worker instance.
    '''

    def make_handler(coord):
        class TrainingCoordinationHandler(BaseHTTPServer.BaseHTTPRequestHandler):
            '''Handles HTTP requests from remote workers to the Training Coordinator.
            '''
            def _send_answer(self, data=None):
                self.send_response(200)
                self.send_header('content-type', 'text/plain')
                self.end_headers()
                if data:
                    self.wfile.write(data)

            def do_GET(self):
                if coord.started:
                    if self.path.startswith(PREFIX_NEXT_INDEX):
                        index = coord.get_next_index(self.path[len(PREFIX_NEXT_INDEX):])
                        if index >= 0:
                            self._send_answer(str(index).encode("utf-8"))
                            return
                    elif self.path.startswith(PREFIX_GET_JOB):
                        job = coord.get_job(worker=int(self.path[len(PREFIX_GET_JOB):]))
                        if job:
                            self._send_answer(pickle.dumps(job))
                            return
                    self.send_response(204) # end of training
                else:
                    self.send_response(202) # not ready yet
                self.end_headers()

            def do_POST(self):
                if coord.started:
                    src = self.rfile.read(int(self.headers['content-length']))
                    job = coord.next_job(pickle.loads(src))
                    if job:
                        self._send_answer(pickle.dumps(job))
                        return
                    self.send_response(204) # end of training
                else:
                    self.send_response(202) # not ready yet
                self.end_headers()

            def log_message(self, format, *args):
                '''Overriding base method to suppress web handler messages on stdout.
                '''
                return

        return TrainingCoordinationHandler

    def __init__(self, is_chief):
        self._init()
        self._lock = Lock()
        self._thread = None
        self.started = False
        self.is_chief = is_chief
        if is_chief:
            self._httpd = BaseHTTPServer.HTTPServer((FLAGS.coord_host, FLAGS.coord_port), TrainingCoordinator.make_handler(self))

    def _reset_counters(self):
        self._index_train = 0
        self._index_dev = 0

    def _init(self):
        self._epochs_running = []
        self._epochs_done = []
        self._reset_counters()
        self._dev_losses = []

    def _log_all_jobs(self):
        '''Use this to debug-print epoch state'''
        log_debug('Epochs - running: %d, done: %d' % (len(self._epochs_running), len(self._epochs_done)))
        for epoch in self._epochs_running:
            log_debug('       - running: ' + epoch.job_status())

    def start_coordination(self, model_feeder, step=0):
        '''Starts to coordinate epochs and jobs among workers on base of
        data-set sizes, the (global) step and FLAGS parameters.

        Args:
            model_feeder (ModelFeeder): data-sets to be used for coordinated training

        Kwargs:
            step (int): global step of a loaded model to determine starting point
        '''
        with self._lock:
            self._init()

            # Number of GPUs per worker - fixed for now by local reality or cluster setup
            gpus_per_worker = len(Config.available_devices)

            # Number of batches processed per job per worker
            batches_per_job  = gpus_per_worker * max(1, FLAGS.iters_per_worker)

            # Number of batches per global step
            batches_per_step = gpus_per_worker * max(1, FLAGS.replicas_to_agg)

            # Number of global steps per epoch - to be at least 1
            steps_per_epoch = max(1, model_feeder.train.total_batches // batches_per_step)

            # The start epoch of our training
            self._epoch = step // steps_per_epoch

            # Number of additional 'jobs' trained already 'on top of' our start epoch
            jobs_trained = (step % steps_per_epoch) * batches_per_step // batches_per_job

            # Total number of train/dev jobs covering their respective whole sets (one epoch)
            self._num_jobs_train = max(1, model_feeder.train.total_batches // batches_per_job)
            self._num_jobs_dev   = max(1, model_feeder.dev.total_batches   // batches_per_job)

            if FLAGS.epoch < 0:
                # A negative epoch means to add its absolute number to the epochs already computed
                self._target_epoch = self._epoch + abs(FLAGS.epoch)
            else:
                self._target_epoch = FLAGS.epoch

            # State variables
            # We only have to train, if we are told so and are not at the target epoch yet
            self._train = FLAGS.train and self._target_epoch > self._epoch

            if self._train:
                # The total number of jobs for all additional epochs to be trained
                # Will be decremented for each job that is produced/put into state 'open'
                self._num_jobs_train_left = (self._target_epoch - self._epoch) * self._num_jobs_train - jobs_trained
                log_info('STARTING Optimization')
                self._training_time = stopwatch()

            # Important for debugging
            log_debug('step: %d' % step)
            log_debug('epoch: %d' % self._epoch)
            log_debug('target epoch: %d' % self._target_epoch)
            log_debug('steps per epoch: %d' % steps_per_epoch)
            log_debug('number of batches in train set: %d' % model_feeder.train.total_batches)
            log_debug('batches per job: %d' % batches_per_job)
            log_debug('batches per step: %d' % batches_per_step)
            log_debug('number of jobs in train set: %d' % self._num_jobs_train)
            log_debug('number of jobs already trained in first epoch: %d' % jobs_trained)

            self._next_epoch()

        # The coordinator is ready to serve
        self.started = True

    def _next_epoch(self):
        # State-machine of the coordination process

        # Indicates, if there were 'new' epoch(s) provided
        result = False

        # Make sure that early stop is enabled and validation part is enabled
        if (FLAGS.early_stop is True) and (FLAGS.validation_step > 0) and (len(self._dev_losses) >= FLAGS.earlystop_nsteps):

            # Calculate the mean of losses for past epochs
            mean_loss = np.mean(self._dev_losses[-FLAGS.earlystop_nsteps:-1])
            # Calculate the standard deviation for losses from validation part in the past epochs
            std_loss = np.std(self._dev_losses[-FLAGS.earlystop_nsteps:-1])
            # Update the list of losses incurred
            self._dev_losses = self._dev_losses[-FLAGS.earlystop_nsteps:]
            log_debug('Checking for early stopping (last %d steps) validation loss: %f, with standard deviation: %f and mean: %f' % (FLAGS.earlystop_nsteps, self._dev_losses[-1], std_loss, mean_loss))

            # Check if validation loss has started increasing or is not decreasing substantially, making sure slight fluctuations don't bother the early stopping from working
            if self._dev_losses[-1] > np.max(self._dev_losses[:-1]) or (abs(self._dev_losses[-1] - mean_loss) < FLAGS.estop_mean_thresh and std_loss < FLAGS.estop_std_thresh):
                # Time to early stop
                log_info('Early stop triggered as (for last %d steps) validation loss: %f with standard deviation: %f and mean: %f' % (FLAGS.earlystop_nsteps, self._dev_losses[-1], std_loss, mean_loss))
                self._dev_losses = []
                self._end_training()
                self._train = False

        if self._train:
            # We are in train mode
            if self._num_jobs_train_left > 0:
                # There are still jobs left
                num_jobs_train = min(self._num_jobs_train_left, self._num_jobs_train)
                self._num_jobs_train_left -= num_jobs_train

                # Let's try our best to keep the notion of curriculum learning
                self._reset_counters()

                # Append the training epoch
                self._epochs_running.append(Epoch(self, self._epoch, num_jobs_train, set_name='train'))

                if FLAGS.validation_step > 0 and (FLAGS.validation_step == 1 or self._epoch > 0) and self._epoch % FLAGS.validation_step == 0:
                    # The current epoch should also have a validation part
                    self._epochs_running.append(Epoch(self, self._epoch, self._num_jobs_dev, set_name='dev'))


                # Indicating that there were 'new' epoch(s) provided
                result = True
            else:
                # No jobs left, but still in train mode: concluding training
                self._end_training()
                self._train = False

        if result:
            # Increment the epoch index
            self._epoch += 1
        return result

    def _end_training(self):
        self._training_time = stopwatch(self._training_time)
        log_info('FINISHED Optimization - training time: %s' % format_duration(self._training_time))

    def start(self):
        '''Starts Training Coordinator. If chief, it starts a web server for
        communication with non-chief instances.
        '''
        if self.is_chief:
            log_debug('Starting coordinator...')
            self._thread = Thread(target=self._httpd.serve_forever)
            self._thread.daemon = True
            self._thread.start()
            log_debug('Coordinator started. Thread id {}'.format(self._thread.ident))

    def stop(self, wait_for_running_epochs=True):
        '''Stops Training Coordinator. If chief, it waits for all epochs to be
        'done' and then shuts down the web server.
        '''
        if self.is_chief and self._thread:
            if wait_for_running_epochs:
                while len(self._epochs_running) > 0:
                    log_traffic('Coordinator is waiting for epochs to finish...')
                    time.sleep(5)
            log_debug('Stopping coordinator...')
            self._httpd.shutdown()
            log_debug('Coordinator stopped.')

    def _talk_to_chief(self, path, data=None, default=None):
        tries = 0
        while tries < FLAGS.coord_retries:
            tries += 1
            try:
                url = 'http://%s:%d%s' % (FLAGS.coord_host, FLAGS.coord_port, path)
                log_traffic('Contacting coordinator - url: %s, tries: %d ...' % (url, tries-1))
                res = urllib.request.urlopen(urllib.request.Request(url, data, { 'content-type': 'text/plain' }))
                str = res.read()
                status = res.getcode()
                log_traffic('Coordinator responded - url: %s, status: %s' % (url, status))
                if status == 200:
                    return str
                if status == 204: # We use 204 (no content) to indicate end of training
                    return default
            except urllib.error.HTTPError as error:
                log_traffic('Problem reaching coordinator - url: %s, HTTP code: %d' % (url, error.code))
                pass
            time.sleep(10)
        return default

    def get_next_index(self, set_name):
        '''Retrives a new cluster-unique batch index for a given set-name.
        Prevents applying one batch multiple times per epoch.

        Args:
            set_name (str): name of the data set - one of 'train', 'dev'

        Returns:
            int. new data set index
        '''
        with self._lock:
            if self.is_chief:
                member = '_index_' + set_name
                value = getattr(self, member, -1)
                setattr(self, member, value + 1)
                return value
            else:
                # We are a remote worker and have to hand over to the chief worker by HTTP
                log_traffic('Asking for next index...')
                value = int(self._talk_to_chief(PREFIX_NEXT_INDEX + set_name))
                log_traffic('Got index %d.' % value)
                return value

    def _get_job(self, worker=0):
        job = None
        # Find first running epoch that provides a next job
        for epoch in self._epochs_running:
            job = epoch.get_job(worker)
            if job:
                return job
        # No next job found
        return None

    def get_job(self, worker=0):
        '''Retrieves the first job for a worker.

        Kwargs:
            worker (int): index of the worker to get the first job for

        Returns:
            WorkerJob. a job of one of the running epochs that will get
                associated with the given worker and put into state 'running'
        '''
        # Let's ensure that this does not interfere with other workers/requests
        with self._lock:
            if self.is_chief:
                # First try to get a next job
                job = self._get_job(worker)

                if job is None:
                    # If there was no next job, we give it a second chance by triggering the epoch state machine
                    if self._next_epoch():
                        # Epoch state machine got a new epoch
                        # Second try to get a next job
                        job = self._get_job(worker)
                        if job is None:
                            # Albeit the epoch state machine got a new epoch, the epoch had no new job for us
                            log_error('Unexpected case - no job for worker %d.' % (worker))
                        return job

                    # Epoch state machine has no new epoch
                    # This happens at the end of the whole training - nothing to worry about
                    log_traffic('No jobs left for worker %d.' % (worker))
                    self._log_all_jobs()
                    return None

                # We got a new job from one of the currently running epochs
                log_traffic('Got new %s' % job)
                return job

            # We are a remote worker and have to hand over to the chief worker by HTTP
            result = self._talk_to_chief(PREFIX_GET_JOB + str(FLAGS.task_index))
            if result:
                result = pickle.loads(result)
            return result

    def next_job(self, job):
        '''Sends a finished job back to the coordinator and retrieves in exchange the next one.

        Kwargs:
            job (WorkerJob): job that was finished by a worker and who's results are to be
                digested by the coordinator

        Returns:
            WorkerJob. next job of one of the running epochs that will get
                associated with the worker from the finished job and put into state 'running'
        '''
        if self.is_chief:
            # Try to find the epoch the job belongs to
            epoch = next((epoch for epoch in self._epochs_running if epoch.id == job.epoch_id), None)
            if epoch:
                # We are going to manipulate things - let's avoid undefined state
                with self._lock:
                    # Let the epoch finish the job
                    epoch.finish_job(job)
                    # Check, if epoch is done now
                    if epoch.done():
                        # If it declares itself done, move it from 'running' to 'done' collection
                        self._epochs_running.remove(epoch)
                        self._epochs_done.append(epoch)
                        log_info('%s' % epoch)
            else:
                # There was no running epoch found for this job - this should never happen.
                log_error('There is no running epoch of ID %d for job with ID %d. This should never happen.' % (job.epoch_id, job.id))
            return self.get_job(job.worker)

        # We are a remote worker and have to hand over to the chief worker by HTTP
        result = self._talk_to_chief('', data=pickle.dumps(job))
        if result:
            result = pickle.loads(result)
        return result
