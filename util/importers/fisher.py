from __future__ import print_function
from __future__ import absolute_import
import fnmatch
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
from threading import Thread
from util.audio import audiofile_to_input_vector
from util.gpu import get_available_gpus
from util.data_set_helpers import DataSets
from util.text import text_to_char_array, validate_label, ctc_label_dense_to_sparse
from six.moves import range

class DataSet(object):
    def __init__(self, txt_files, thread_count, batch_size, numcep, numcontext, next_index=lambda x: x + 1):
        self._coord = None
        self._numcep = numcep
        self._x = tf.placeholder(tf.float32, [None, numcep + (2 * numcep * numcontext)])
        self._x_length = tf.placeholder(tf.int32, [])
        self._y = tf.placeholder(tf.int32, [None,])
        self._y_length = tf.placeholder(tf.int32, [])
        self.example_queue = tf.PaddingFIFOQueue(shapes=[[None, numcep + (2 * numcep * numcontext)], [], [None,], []],
                                                  dtypes=[tf.float32, tf.int32, tf.int32, tf.int32],
                                                  capacity=2 * self._get_device_count() * batch_size)
        self._enqueue_op = self.example_queue.enqueue([self._x, self._x_length, self._y, self._y_length])
        self._close_op = self.example_queue.close(cancel_pending_enqueues=True)
        self._txt_files = txt_files
        self.batch_size = batch_size
        self._numcontext = numcontext
        self._thread_count = thread_count
        self._files_list = self._create_files_list()
        self._next_index = next_index

    def _get_device_count(self):
        available_gpus = get_available_gpus()
        return max(len(available_gpus), 1)

    def start_queue_threads(self, session, coord):
        self._coord = coord
        batch_threads = [Thread(target=self._populate_batch_queue, args=(session,)) for i in range(self._thread_count)]
        for batch_thread in batch_threads:
            batch_thread.daemon = True
            batch_thread.start()
        return batch_threads

    def close_queue(self, session):
        session.run(self._close_op)

    def _create_files_list(self):
        priorityQueue = PriorityQueue()
        for txt_file in self._txt_files:
            wav_file = os.path.splitext(txt_file)[0] + ".wav"
            wav_file_size = os.path.getsize(wav_file)
            priorityQueue.put((wav_file_size, (txt_file, wav_file)))
        files_list = []
        while not priorityQueue.empty():
            priority, (txt_file, wav_file) = priorityQueue.get()
            files_list.append((txt_file, wav_file))
        return files_list

    def _indices(self):
        index = -1
        while not self._coord.should_stop():
            index = self._next_index(index) % len(self._files_list)
            yield self._files_list[index]

    def _populate_batch_queue(self, session):
        for txt_file, wav_file in self._indices():
            source = audiofile_to_input_vector(wav_file, self._numcep, self._numcontext)
            source_len = len(source)
            with codecs.open(txt_file, encoding="utf-8") as open_txt_file:
                # We need to do the encode-decode dance here because encode
                # returns a bytes() object on Python 3, and text_to_char_array
                # expects a string.
                target = unicodedata.normalize("NFKD", open_txt_file.read())   \
                                    .encode("ascii", "ignore")                 \
                                    .decode("ascii", "ignore")
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
        source, source_lengths, target, target_lengths = self.example_queue.dequeue_many(self.batch_size)
        sparse_labels = ctc_label_dense_to_sparse(target, target_lengths, self.batch_size)
        return source, source_lengths, sparse_labels

    @property
    def total_batches(self):
        # Note: If len(_txt_files) % batch_size != 0, this re-uses initial _txt_files
        return int(ceil(float(len(self._txt_files)) /float(self.batch_size)))

def read_data_sets(data_dir, train_batch_size, dev_batch_size, test_batch_size, numcep, numcontext, thread_count=8, stride=1, offset=0, next_index=lambda s, i: i + 1, limit_dev=0, limit_test=0, limit_train=0, sets=[]):
    # Assume data_dir contains extracted LDC2004S13, LDC2004T19, LDC2005S13, LDC2005T19

    # Conditionally convert Fisher sph data to wav
    _maybe_convert_wav(data_dir, "LDC2004S13", "fisher-2004-wav")
    _maybe_convert_wav(data_dir, "LDC2005S13", "fisher-2005-wav")

    # Conditionally split Fisher wav data
    _maybe_split_wav(data_dir, os.path.join("LDC2004T19", "fe_03_p1_tran", "data", "trans"), "fisher-2004-wav", "fisher-2004-split-wav")
    _maybe_split_wav(data_dir, os.path.join("LDC2005T19", "fe_03_p2_tran", "data", "trans"), "fisher-2005-wav", "fisher-2005-split-wav")

    # Conditionally split Fisher transcriptions
    _maybe_split_transcriptions(data_dir, os.path.join("LDC2004T19", "fe_03_p1_tran", "data", "trans"), "fisher-2004-split-wav")
    _maybe_split_transcriptions(data_dir, os.path.join("LDC2005T19", "fe_03_p2_tran", "data", "trans"), "fisher-2005-split-wav")

    # Conditionally split Fisher data into train/validation/test sets
    _maybe_split_sets(data_dir, "fisher-2004-split-wav", "fisher-2004-split-wav-sets")
    _maybe_split_sets(data_dir, "fisher-2005-split-wav", "fisher-2005-split-wav-sets")

    # The following file has an incorrect transcript that is much longer than
    # the audio source. The result is that we end up with more labels than time
    # slices, which breaks CTC. We fix this directly since it's a single occurrence
    # in the entire corpus.
    problematic_file = os.path.join(data_dir, "fisher-2004-split-wav-sets", "train", "fe_03_00265-33.53-33.81.txt")
    with open(problematic_file, "w") as f:
        f.write("correct")

    # Create train DataSet
    train = None
    if "train" in sets:
        train = _read_data_set(data_dir, "fisher-200?-split-wav-sets/train", thread_count, train_batch_size, numcep, numcontext, stride=stride, offset=offset, next_index=lambda i: next_index('train', i), limit=limit_train)

    # Create dev DataSet
    dev = None
    if "dev" in sets:
        dev = _read_data_set(data_dir, "fisher-200?-split-wav-sets/dev", thread_count, dev_batch_size, numcep, numcontext, stride=stride, offset=offset, next_index=lambda i: next_index('dev', i), limit=limit_dev)

    # Create test DataSet
    test = None
    if "test" in sets:
        test = _read_data_set(data_dir, "fisher-200?-split-wav-sets/test", thread_count, test_batch_size, numcep, numcontext, stride=stride, offset=offset, next_index=lambda i: next_index('test', i), limit=limit_test)

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
            sph_file = os.path.join(root, filename)
            for channel in ["1", "2"]:
                wav_filename = os.path.splitext(os.path.basename(sph_file))[0] + "_c" + channel + ".wav"
                wav_file = os.path.join(target_dir, wav_filename)
                print("converting {} to {}".format(sph_file, wav_file))
                subprocess.check_call(["sph2pipe", "-c", channel, "-p", "-f", "rif", sph_file, wav_file])

def _parse_transcriptions(trans_file):
    segments = []
    with open(trans_file, "r") as fin:
        for line in fin:
            if line.startswith("#") or len(line) <= 1:
                continue

            start_time_beg = 0
            start_time_end = line.find(" ", start_time_beg)

            stop_time_beg = start_time_end + 1
            stop_time_end = line.find(" ", stop_time_beg)

            speaker_beg = stop_time_end + 1
            speaker_end = line.find(" ", speaker_beg)

            transcript_beg = speaker_end + 1
            transcript_end = len(line)

            segments.append({
                "start_time": float(line[start_time_beg:start_time_end]),
                "stop_time": float(line[stop_time_beg:stop_time_end]),
                "speaker": line[speaker_beg:speaker_end],
                "transcript": line[transcript_beg:transcript_end].strip(),
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
        for filename in fnmatch.filter(filenames, "*.txt"):
            trans_file = os.path.join(root, filename)
            segments = _parse_transcriptions(trans_file)

            # Open wav corresponding to transcription file
            wav_filenames = [os.path.splitext(os.path.basename(trans_file))[0] + "_c" + channel + ".wav" for channel in ["1", "2"]]
            wav_files = [os.path.join(source_dir, wav_filename) for wav_filename in wav_filenames]

            print("splitting {} according to {}".format(wav_files, trans_file))

            origAudios = [wave.open(wav_file, "r") for wav_file in wav_files]

            # Loop over segments and split wav_file for each segment
            for segment in segments:
                # Create wav segment filename
                start_time = segment["start_time"]
                stop_time = segment["stop_time"]
                new_wav_filename = os.path.splitext(os.path.basename(trans_file))[0] + "-" + str(start_time) + "-" + str(stop_time) + ".wav"
                new_wav_file = os.path.join(target_dir, new_wav_filename)

                channel = 0 if segment["speaker"] == "A:" else 1
                _split_wav(origAudios[channel], start_time, stop_time, new_wav_file)

            # Close origAudios
            for origAudio in origAudios:
                origAudio.close()

def _split_wav(origAudio, start_time, stop_time, new_wav_file):
    frameRate = origAudio.getframerate()
    origAudio.setpos(int(start_time*frameRate))
    chunkData = origAudio.readframes(int((stop_time - start_time)*frameRate))
    chunkAudio = wave.open(new_wav_file, "w")
    chunkAudio.setnchannels(origAudio.getnchannels())
    chunkAudio.setsampwidth(origAudio.getsampwidth())
    chunkAudio.setframerate(frameRate)
    chunkAudio.writeframes(chunkData)
    chunkAudio.close()

def _maybe_split_transcriptions(data_dir, original_data, converted_data):
    source_dir = os.path.join(data_dir, original_data)
    target_dir = os.path.join(data_dir, converted_data)

    if os.path.exists(os.path.join(source_dir, "split_transcriptions_done")):
        print("skipping maybe_split_transcriptions")
        return

    # Loop over transcription files and split them into individual files for
    # each utterance
    for root, dirnames, filenames in os.walk(source_dir):
        for filename in fnmatch.filter(filenames, "*.txt"):
            trans_file = os.path.join(root, filename)
            segments = _parse_transcriptions(trans_file)

            # Loop over segments and split wav_file for each segment
            for segment in segments:
                start_time = segment["start_time"]
                stop_time = segment["stop_time"]
                txt_filename = os.path.splitext(os.path.basename(trans_file))[0] + "-" + str(start_time) + "-" + str(stop_time) + ".txt"
                txt_file = os.path.join(target_dir, txt_filename)

                transcript = validate_label(segment["transcript"])

                # If the transcript is valid, write it to the segment file
                if transcript != None:
                    with open(txt_file, "w") as fout:
                        fout.write(transcript)

    with open(os.path.join(source_dir, "split_transcriptions_done"), "w") as fout:
        fout.write("This file signals to the importer than the transcription of this source dir has already been completed.")

def _maybe_split_sets(data_dir, original_data, converted_data):
    source_dir = os.path.join(data_dir, original_data)
    target_dir = os.path.join(data_dir, converted_data)

    filelist = sorted(glob(os.path.join(source_dir, "*.txt")))

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

def _read_data_set(work_dir, data_set, thread_count, batch_size, numcep, numcontext, stride=1, offset=0, next_index=lambda i: i + 1, limit=0):
    # Create data set dir
    dataset_dir = os.path.join(work_dir, data_set)

    # Obtain list of txt files
    txt_files = glob(os.path.join(dataset_dir, "*.txt"))
    if limit > 0:
        txt_files = txt_files[:limit]
    txt_files = txt_files[offset::stride]

    # Return DataSet
    return DataSet(txt_files, thread_count, batch_size, numcep, numcontext, next_index=next_index)
