import os
import re
import csv
import glob
import tensorflow as tf
import time
from six.moves import range
from util.log import Logger

log = Logger(id='persistence', caption='Persistence')

class CheckpointManager(object):
    '''
    A class to manage checkpointing, restoring and initialization of a model.
    It also manages filenames of two checkpoint series/families (epoch ones and intermediate ones)
    within a checkpoint directory.
    '''
    def __init__(self, checkpoint_dir=None, load='recent', inter_secs=0, keep_n_inters=3, keep_n_epochs=5, init_op=None, saver=None):
        '''
        Initializes a new checkpoint manager.
        'checkpoint_dir' - path of the checkpoint directory - if omitted, checkpointing will be deactivated.
        'load' - one of
         - 'recent' (default) to load the most recent checkpoint,
         - 'best-dev' for loading the existing validated epoch with the lowest validation loss
         - 'last-epoch' for loading the epoch checkpoint with the highest number
         - <filename> for directly loading a checkpoint from the given filename
         - 'init' (also fallback if no checkpoints were found) for initializing a new model
        'inter_secs' - time interval in seconds for saving intermediate checkpoints - 0 deactivates intermediate checkpointing
        'keep_n_inters' - how many intermediate checkpoints to keep - 0 deactivates intermediate checkpointing
        'keep_n_epochs' - how many epoch checkpoints to keep - 0 deactivates epoch checkpointing
        'init_op' - optional initialization operation for the model (if omitted, it will use standard global initialization)
        'saver' - optional saver for the model (if omitted, it will use an internal one)
        '''
        self.checkpoint_dir = checkpoint_dir
        self.load = load
        self.inter_secs = inter_secs
        self.keep_n_inters = keep_n_inters
        self.keep_n_epochs = keep_n_epochs
        self.init_op = init_op if init_op else tf.global_variables_initializer()
        self.saver = saver if saver else tf.train.Saver()
        self._t0 = None
        self._results = []
        if checkpoint_dir:
            self._csv = os.path.join(checkpoint_dir, 'results.csv')
            if os.path.isfile(self._csv):
                with open(self._csv) as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        self._results.append((int(row['epoch']), float(row['loss'], float(row['dev-loss']))
                # ordered by epoch
                self._results = sorted(self._results, key=lambda r: r[0])

    def _get_checkpoints(self):
        '''
        Scans checkpoint dir and returns a list of checkpoint tuples
        ordered by their global index.
        Each checkpoint tuple consists of the
         - epoch index (1-based, 0 for intermediate checkpoints),
         - global index and a
         - set of filenames.
        '''
        # list and hash for collecting checkpoint tuples
        checkpoints = []
        checkpoints_lookup = {}
        # scanning checkpoint dir for checkpoints...
        for filename in glob.glob(self.checkpoint_dir + '/*'):
            # only process filenames of the following form
            found = re.search('(inter|epoch-[0-9]+)_([0-9]+)\\.ckpt.*', filename, re.IGNORECASE)
            if not found:
                continue
            # extract the epoch index - 0 for intermediate checkpoints
            epoch_index = found.group(1)
            epoch_index = int(epoch_index[6:]) if epoch_index.startswith('epoch-') else 0
            # extract the global index (underscore part)
            global_index = int(found.group(2))
            # for looking up checkpoint tuples within this loop
            key = '%d-%d' % (epoch_index, global_index)
            if key in checkpoints_lookup:
                # we already processed some file(s) for this checkpoint...
                _, _, filenames = checkpoints_lookup[key]
            else:
                # creating a new checkpoint tuple
                filenames = []
                checkpoint = (epoch_index, global_index, filenames)
                checkpoints_lookup[key] = checkpoint
                checkpoints.append(checkpoint)
            # add current filename to checkpoint's file-set
            filenames.append(filename)
        # return list of checkpoint tuples - ordered by their global index
        return sorted(checkpoints, key=lambda cp: cp[1])

    def _get_filename(self, filenames):
        '''
        Strips the filename-suffix from first file of the passed checkpoint file-set
        to get a common prefix that can be used to restore the checkpoint.
        '''
        return '.'.join(filenames[0].split('.')[:-1])

    def _delete_files(self, filenames):
        '''
        Deletes files of a file-set.
        '''
        for filename in filenames:
            log.step('Removing file "%s"...' % filename)
            os.remove(filename)

    def _prune_and_save(self, session, epoch_index, global_index):
        '''
        Removes outdated checkpoint files and stores a new checkpoint.
        '''
        keep = self.keep_n_inters if epoch_index == 0 else self.keep_n_epochs
        assert keep > 0
        checkpoints = []
        for checkpoint in self._get_checkpoints():
            # if checkpoint's global index is greater than this one,
            if checkpoint[1] > global_index:
                # it is on an outdated branch of training history and will be deleted
                log.debug('Removing files of checkpoint "%s", as the global index %d is greater than the new one (%d)...' % \
                    (self._get_filename(filenames), checkpoint[1], global_index))
                self._delete_files(filenames)
            else:
                # for the next steps we only keep checkpoints of the current
                # checkpoint family (either epoch ones or intermediate ones)
                if (epoch_index == 0 and checkpoint[0] == 0) or (epoch_index > 0 and checkpoint[0] > 0):
                    checkpoints.append(checkpoint)
        # removing checkpoint files within current checkpoint family that fall off the keep range
        for _, _, filenames in checkpoints[:-(keep - 1)]:
            log.debug('Removing files of "%s", as only %d checkpoints should be kept...' % \
                (self._get_filename(filenames), keep))
            self._delete_files(filenames)
        # generating checkpoint filename
        prefix = ('epoch-%d' % epoch_index) if epoch_index > 0 else 'inter'
        filename = '%s_%d.ckpt' % (prefix, global_index)
        filename = os.path.join(self.checkpoint_dir, filename)
        log.info('Writing checkpoint "%s"...' % filename)
        # finally: saving the new checkpoint
        self.saver.save(session, filename, write_state=False)

    def _init(self, session):
        '''
        Initializes the session.
        '''
        log.info('No checkpoint to load - initializing new model...')
        session.run(self.init_op)

    def _restore(self, session, filename):
        '''
        Restores a checkpoint file-set.
        '''
        log.info('Restoring checkpoint "%s"...' % filename)
        self.saver.restore(session, filename)

    def start(self, session):
        '''
        Starts a training session by either restoring the right checkpoint
        or by initializing the model.
        Should be called at the beginning of a training session.
        '''
        if not self.checkpoint_dir:
            self._init(session)
            return
        checkpoint = None
        # collect available checkpoints
        checkpoints = self._get_checkpoints()
        log.debug('Found %d checkpoints in directory "%s".' % (len(checkpoints), self.checkpoint_dir))
        if self.load == 'recent':
            if len(checkpoints) > 0:
                # pick the most recent checkpoint (by means of global-index),
                # no matter if this is an epoch or intermediate one
                checkpoint = checkpoints[-1]
        else:
            # epoch checkpoints only, ordered by epoch number
            checkpoints = sorted([cp for cp in checkpoints if cp[0] > 0], key=lambda cp: cp[0])
            if len(checkpoints) > 0:
                if self.load == 'best-dev':
                    # get epoch numbers of existing checkpoints
                    available = [cp[0] for cp in checkpoints]
                    # and intersect them with result-file-listed epochs featuring a dev-loss
                    available = [r for r in self._results if r[0] in available and not r[2] is None]
                    # if there are any,
                    if len(available) > 0:
                        # pick the one with dev-loss minimum
                        epoch = min(available, key=lambda r: r[2])[0]
                        checkpoint = next(cp for cp in checkpoints if cp[0] == epoch)
                elif self.load == 'last-epoch':
                    # pick the one with the highest epoch number
                    checkpoint = checkpoints[-1]
        if checkpoint:
            # so we got a checkpoint file-set - restoring it...
            self._restore(session, self._get_filename(checkpoint[2]))
        elif self.load != 'init' and len(glob.glob(self.load + '*')) > 0:
            # if self.load is not a keyword, but a path-prefix to existing files, we restore it...
            self._restore(session, self.load)
        else:
            # final fallback -> fresh start (also self.load == 'init')
            self._init(session)
        # setting t0 for intermediate checkpointing
        self._t0 = time.time()
        # return epoch history
        return self._results

    def step(self, session, global_index):
        '''
        Allows checkpoint manager to save an intermediate checkpoint,
        if the internal timer elapsed.
        Should be called after each trained batch within a training loop.
        'global_index' should represent the overall training progress
        (one could use TF's global_step or the total number of applied samples).
        '''
        # no intermediate checkpointing, if no dir or nothing to keep
        if not self.checkpoint_dir or self.keep_n_inters <= 0:
            return
        current_time = time.time()
        if self.inter_secs > 0 and current_time - self._t0 > self.inter_secs:
            # our timer elapsed -> restart timer
            self._t0 = current_time
            # removing outdated intermediate checkpoint files and saving the intermediate checkpoint...
            self._prune_and_save(session, 0, global_index)

    def epoch(self, session, epoch_number, global_index, loss, dev_loss=None):
        '''
        Allows checkpoint manager to save an epoch checkpoint.
        Should be called after each trained epoch (with dev-loss if validated).
        'epoch_number' is the 1-based epoch index.
        'global_index' should represent the overall training progress
        (one could use TF's global_step or the total number of applied samples).
        'loss' should be the epoch's mean training loss.
        'dev_loss' (optional) should be the epoch's mean validation loss.
        '''
        # no epoch checkpointing, if no dir or nothing to keep
        if not self.checkpoint_dir or self.keep_n_epochs <= 0:
            return
        # removing outdated epoch checkpoint files and saving the epoch checkpoint...
        self._prune_and_save(session, epoch_number, global_index)
        log.debug('Updating "%s"...' % self._csv)
        # removing higher epoch numbers from results log, as they got pruned and
        # are on an outdated branch of training history
        self._results = [r for r in self._results if r[0] < epoch_number]
        # append our new epoch
        self._results.append((epoch_number, loss, dev_loss))
        # write out results.csv
        with open(self._csv, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['epoch', 'loss', 'dev-loss'])
            writer.writeheader()
            for r in self._results:
                writer.writerow({ 'epoch': r.epoch, 'loss': r.loss, 'dev-loss': r.dev_loss })

