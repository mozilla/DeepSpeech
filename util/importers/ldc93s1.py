from os import path
from glob import glob
from util.datasets import BaseDataSet, DataSets
from util.text import text_to_char_array
from util.audio import audiofile_to_input_vector
from tensorflow.contrib.learn.python.learn.datasets import base

class DataSet(BaseDataSet):
    def __init__(self, *args, **kwargs):
        super(DataSet, self).__init__(*args, **kwargs)
        self._start_queue_threads()
    
    def _compute_source_target(self):
        txt_file = self._txt_files[0]
        wav_file = path.splitext(txt_file)[0] + ".wav"

        audio_waves = audiofile_to_input_vector(wav_file, self._num_mfcc_features, self._num_context)
        
        with open(txt_file) as open_txt_file:
            original = ' '.join(open_txt_file.read().strip().lower().split(' ')[2:]).replace('.', '')

        target = text_to_char_array(original)

        return audio_waves, len(audio_waves), target, len(target)

    def _populate_batch_queue(self):
        source, source_len, target, target_len = self._compute_source_target()
        while True:
            self._session.run(self._enqueue_op, feed_dict={
                self._x: source,
                self._x_length: source_len,
                self._y: target,
                self._y_length: target_len})

def read_data_sets(session, data_dir, batch_size, numcep, numcontext, thread_count=1):
    # Conditionally download data
    LDC93S1_BASE = "LDC93S1"
    LDC93S1_BASE_URL = "https://catalog.ldc.upenn.edu/desc/addenda/"
    local_file = base.maybe_download(LDC93S1_BASE + ".wav", data_dir, LDC93S1_BASE_URL + LDC93S1_BASE + ".wav")
    _ = base.maybe_download(LDC93S1_BASE + ".txt", data_dir, LDC93S1_BASE_URL + LDC93S1_BASE + ".txt")

    # Create all DataSets, we do not really need separation
    train = dev = test = _read_data_set(session, data_dir, thread_count, batch_size, numcep, numcontext)

    # Return DataSets
    return DataSets(train, dev, test)

def _read_data_set(session, data_dir, thread_count, batch_size, numcep, numcontext):
    # Obtain list of txt files
    txt_files = glob(path.join(data_dir, "*.txt"))

    # Return DataSet
    return DataSet(session, txt_files, thread_count, batch_size, numcep, numcontext)
