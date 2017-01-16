import fnmatch
import numpy as np
import os
import subprocess
import wave
import tensorflow as tf
import unicodedata
import codecs
from glob import glob
from itertools import cycle
from math import ceil
from Queue import PriorityQueue
from Queue import Queue
from threading import Thread
from util.audio import audiofile_to_input_vector
from util.gpu import get_available_gpus
from util.text import text_to_char_array, ctc_label_dense_to_sparse, validate_label

class DataSets(object):
    def __init__(self, train, dev, test):
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


class DataSet(object):
    def __init__(self, txt_files, thread_count, batch_size, numcep, numcontext):
        self._coord = None
        self._numcep = numcep
        self._x = tf.placeholder(tf.float32, [None, numcep + (2 * numcep * numcontext)])
        self._x_length = tf.placeholder(tf.int32, [])
        self._y = tf.placeholder(tf.int32, [None, ])
        self._y_length = tf.placeholder(tf.int32, [])
        self._example_queue = tf.PaddingFIFOQueue(shapes=[[None, numcep + (2 * numcep * numcontext)], [], [None, ], []],
                                                  dtypes = [tf.float32, tf.int32, tf.int32, tf.int32],
                                                  capacity = 2 * self._get_device_count() * batch_size)
        self._enqueue_op = self._example_queue.enqueue([self._x, self._x_length, self._y, self._y_length])
        self._close_op = self._example_queue.close(cancel_pending_enqueues=True)
        self._txt_files = txt_files
        self._batch_size = batch_size
        self._numcontext = numcontext
        self._thread_count = thread_count
        self._files_circular_list = self._create_files_circular_list()

    def _get_device_count(self):
        available_gpus = get_available_gpus()
        return  max(len(available_gpus), 1)

    def start_queue_threads(self, session, coord):
        self._coord = coord
        batch_threads = [Thread(target=self._populate_batch_queue, args=(session,)) for i in xrange(self._thread_count)]
        for batch_thread in batch_threads:
            batch_thread.daemon = True
            batch_thread.start()
        return batch_threads

    def close_queue(self, session):
        session.run(self._close_op)

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

    def _populate_batch_queue(self, session):
        for txt_file, wav_file in self._files_circular_list:
            if self._coord.should_stop():
                return
            source = audiofile_to_input_vector(wav_file, self._numcep, self._numcontext)
            source_len = len(source)
            with codecs.open(txt_file, encoding="utf-8") as open_txt_file:
                target = unicodedata.normalize("NFKD", open_txt_file.read()).encode("ascii", "ignore")
                target = text_to_char_array(target)
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
        source, source_lengths, target, target_lengths = self._example_queue.dequeue_many(self._batch_size)
        sparse_labels = ctc_label_dense_to_sparse(target, target_lengths, self._batch_size)
        return source, source_lengths, sparse_labels

    @property
    def total_batches(self):
        # Note: If len(_txt_files) % _batch_size != 0, this re-uses initial _txt_files
        return int(ceil(float(len(self._txt_files)) / float(self._batch_size)))


def read_data_sets(data_dir, train_batch_size, dev_batch_size, test_batch_size, numcep, numcontext, thread_count=8, limit_dev=0, limit_test=0, limit_train=0):
    data_dir = os.path.join(data_dir, "LDC97S62")

    # Conditionally convert swb sph data to wav
    _maybe_convert_wav(data_dir, "swb1_d1", "swb1_d1-wav")
    _maybe_convert_wav(data_dir, "swb1_d2", "swb1_d2-wav")
    _maybe_convert_wav(data_dir, "swb1_d3", "swb1_d3-wav")
    _maybe_convert_wav(data_dir, "swb1_d4", "swb1_d4-wav")

    # Conditionally split wav data
    _maybe_split_wav(data_dir, "swb_ms98_transcriptions", "swb1_d1-wav",
                     "swb1_d1-split-wav")
    _maybe_split_wav(data_dir, "swb_ms98_transcriptions", "swb1_d2-wav",
                     "swb1_d2-split-wav")
    _maybe_split_wav(data_dir, "swb_ms98_transcriptions", "swb1_d3-wav",
                     "swb1_d3-split-wav")
    _maybe_split_wav(data_dir, "swb_ms98_transcriptions", "swb1_d4-wav",
                     "swb1_d4-split-wav")

    _maybe_split_transcriptions(data_dir, "swb_ms98_transcriptions")

    _maybe_split_sets(data_dir, ["swb1_d1-split-wav", "swb1_d2-split-wav", "swb1_d3-split-wav", "swb1_d4-split-wav"],
                      "final_sets")

    # Create dev DataSet
    dev = _read_data_set(data_dir, "final_sets/dev", thread_count, dev_batch_size, numcep,
                         numcontext, limit_dev)

    # Create test DataSet
    test = _read_data_set(data_dir, "final_sets/test", thread_count, test_batch_size, numcep,
                         numcontext, limit_test)

    # Create train DataSet
    train = _read_data_set(data_dir, "final_sets/train", thread_count, train_batch_size, numcep,
                         numcontext, limit_train)

    # Return DataSets
    return DataSets(train, dev, test)

def _maybe_convert_wav(data_dir, original_data, converted_data):
    source_dir = os.path.join(data_dir, original_data)
    target_dir = os.path.join(data_dir, converted_data)

    # Conditionally convert sph files to wav files
    if os.path.exists(target_dir):
        print("skipping maybe_convert_wav")
        return

    # Create target_dir
    os.makedirs(target_dir)

    # Loop over sph files in source_dir and convert each to 16-bit PCM wav
    for root, dirnames, filenames in os.walk(source_dir):
        for filename in fnmatch.filter(filenames, "*.sph"):
            for channel in ['1', '2']:
                sph_file = os.path.join(root, filename)
                wav_filename = os.path.splitext(os.path.basename(sph_file))[0] + "-" + channel + ".wav"
                wav_file = os.path.join(target_dir, wav_filename)
                print("converting {} to {}".format(sph_file, wav_file))
                subprocess.check_call(["sph2pipe", "-c", channel, "-p", "-f", "rif", sph_file, wav_file])

def _parse_transcriptions(trans_file):
    segments = []
    with open(trans_file, "r") as fin:
        for line in fin:
            if line.startswith("#")  or len(line) <= 1:
                continue

            filename_time_beg = 0;
            filename_time_end = line.find(" ", filename_time_beg)

            start_time_beg = filename_time_end + 1
            start_time_end = line.find(" ", start_time_beg)

            stop_time_beg = start_time_end + 1
            stop_time_end = line.find(" ", stop_time_beg)

            transcript_beg = stop_time_end + 1
            transcript_end = len(line)

            if validate_label(line[transcript_beg:transcript_end].strip()) == None:
                continue

            segments.append({
                "start_time": float(line[start_time_beg:start_time_end]),
                "stop_time": float(line[stop_time_beg:stop_time_end]),
                "speaker": line[6],
                "transcript": line[transcript_beg:transcript_end].strip().lower(),
            })
    return segments


def _maybe_split_wav(data_dir, trans_data, original_data, converted_data):
    trans_dir = os.path.join(data_dir, trans_data)
    source_dir = os.path.join(data_dir, original_data)
    target_dir = os.path.join(data_dir, converted_data)
    if os.path.exists(target_dir):
        print("skipping maybe_split_wav")
        return

    os.makedirs(target_dir)

    # Loop over transcription files and split corresponding wav
    for root, dirnames, filenames in os.walk(trans_dir):
        for filename in fnmatch.filter(filenames, "*.text"):
            if "trans" not in filename:
                continue
            trans_file = os.path.join(root, filename)
            segments = _parse_transcriptions(trans_file)

            # Open wav corresponding to transcription file
            channel = ("2","1")[(os.path.splitext(os.path.basename(trans_file))[0])[6] == 'A']
            wav_filename = "sw0" + (os.path.splitext(os.path.basename(trans_file))[0])[2:6] + "-" + channel + ".wav"
            wav_file = os.path.join(source_dir, wav_filename)

            print("splitting {} according to {}".format(wav_file, trans_file))

            if not os.path.exists(wav_file):
                print("skipping. does not exist:" + wav_file)
                continue

            origAudio = wave.open(wav_file, "r")

            # Loop over segments and split wav_file for each segment
            for segment in segments:
                # Create wav segment filename
                start_time = segment["start_time"]
                stop_time = segment["stop_time"]
                new_wav_filename = os.path.splitext(os.path.basename(trans_file))[0] + "-" + str(
                    start_time) + "-" + str(stop_time) + ".wav"
                new_wav_file = os.path.join(target_dir, new_wav_filename)
                _split_wav(origAudio, start_time, stop_time, new_wav_file)

            # Close origAudio
            origAudio.close()

def _split_wav(origAudio, start_time, stop_time, new_wav_file):
    frameRate = origAudio.getframerate()
    origAudio.setpos(int(start_time * frameRate))
    chunkData = origAudio.readframes(int((stop_time - start_time) * frameRate))
    chunkAudio = wave.open(new_wav_file, "w")
    chunkAudio.setnchannels(origAudio.getnchannels())
    chunkAudio.setsampwidth(origAudio.getsampwidth())
    chunkAudio.setframerate(frameRate)
    chunkAudio.writeframes(chunkData)
    chunkAudio.close()


def _maybe_split_transcriptions(data_dir, original_data):
    source_dir = os.path.join(data_dir, original_data)
    wav_dirs = ["swb1_d1-split-wav", "swb1_d2-split-wav", "swb1_d3-split-wav", "swb1_d4-split-wav"]

    if os.path.exists(os.path.join(source_dir, "split_transcriptions_done")):
        print("skipping maybe_split_transcriptions")
        return

    # Loop over transcription files and split them into individual files for
    # each utterance
    for root, dirnames, filenames in os.walk(source_dir):
        for filename in fnmatch.filter(filenames, "*.text"):
            if "trans" not in filename:
                continue

            trans_file = os.path.join(root, filename)
            segments = _parse_transcriptions(trans_file)

            # Loop over segments and split wav_file for each segment
            for segment in segments:
                start_time = segment["start_time"]
                stop_time = segment["stop_time"]
                txt_filename = os.path.splitext(os.path.basename(trans_file))[0] + "-" + str(start_time) + "-" + str(
                    stop_time) + ".txt"
                wav_filename = os.path.splitext(os.path.basename(trans_file))[0] + "-" + str(start_time) + "-" + str(
                    stop_time) + ".wav"

                transcript = validate_label(segment["transcript"])

                for wav_dir in wav_dirs:
                    if os.path.exists(os.path.join(data_dir, wav_dir, wav_filename)):
                        # If the transcript is valid and the txt segment filename does
                        # not exist create it
                        txt_file = os.path.join(data_dir, wav_dir, txt_filename)
                        if transcript != None and not os.path.exists(txt_file):
                            with open(txt_file, "w") as fout:
                                fout.write(transcript)
                        break

    with open(os.path.join(source_dir, "split_transcriptions_done"), "w") as fout:
        fout.write(
            "This file signals to the importer than the transcription of this source dir has already been completed.")


def _maybe_split_sets(data_dir, original_data, converted_data):
    target_dir = os.path.join(data_dir, converted_data)

    if os.path.exists(target_dir):
        return;

    os.makedirs(target_dir)

    filelist = []
    for dir in original_data:
        source_dir = os.path.join(data_dir, dir)
        filelist.extend(glob(os.path.join(source_dir, "*.txt")))

    # We initially split the entire set into 80% train and 20% test, then
    # split the train set into 80% train and 20% validation.
    train_beg = 0
    train_end = int(0.8 * len(filelist))

    dev_beg = int(0.8 * train_end)
    dev_end = train_end
    train_end = dev_beg

    test_beg = dev_end
    test_end = len(filelist)

    _maybe_split_dataset(filelist[train_beg:train_end], os.path.join(target_dir, "train"))
    _maybe_split_dataset(filelist[dev_beg:dev_end], os.path.join(target_dir, "dev"))
    _maybe_split_dataset(filelist[test_beg:test_end], os.path.join(target_dir, "test"))


def _maybe_split_dataset(filelist, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        for txt_file in filelist:
            new_txt_file = os.path.join(target_dir, os.path.basename(txt_file))
            os.rename(txt_file, new_txt_file)

            wav_file = os.path.splitext(txt_file)[0] + ".wav"
            new_wav_file = os.path.join(target_dir, os.path.basename(wav_file))
            os.rename(wav_file, new_wav_file)


def _read_data_set(work_dir, data_set, thread_count, batch_size, numcep, numcontext, limit=0):
    # Obtain list of txt files
    txt_files = glob(os.path.join(work_dir, data_set, "*.txt"))
    if limit > 0:
        txt_files = txt_files[:limit]

    # Return DataSet
    return DataSet(txt_files, thread_count, batch_size, numcep, numcontext)
