import numpy as np

from os import path
from util.text import text_to_sparse_tuple 
from util.audio import audiofiles_to_audio_data_sets

class DataSets(object):
    def __init__(self, train, validation, test):
        self._train = train
        self._validation = validation
        self._test = test

    @property
    def train(self):
        return self._train

    @property
    def validation(self):
        return self._validation

    @property
    def test(self):
        return self._test

class DataSet(object):
    def __init__(self, inputs, outputs, seq_len):
        self._offset = 0
        self._inputs = inputs
        self._outputs = outputs
        self._seq_len = seq_len

    def next_batch(self, batch_size):
        next_batch = (self._inputs, self._outputs, self._seq_len) # TODO: Choose only batch_size elements
        self._offset += batch_size
        return next_batch

    @property
    def max_batch_seq_len(self):
        return np.amax(self._seq_len)

    @property
    def num_examples(self):
        return self._inputs.shape[0]


def read_data_sets(data_dir, numcep, numcontext):
    # Get train data
    train_outputs = read_text_data_sets(data_dir, 'train')
    train_inputs, train_seq_len = read_audio_data_sets(data_dir, numcep, numcontext, 'train')
    # Get validation data
    validation_outputs = read_text_data_sets(data_dir, 'validation')
    validation_inputs, validation_seq_len = read_audio_data_sets(data_dir, numcep, numcontext, 'validation')
    # Get test data
    test_outputs = read_text_data_sets(data_dir, 'test')
    test_inputs, test_seq_len = read_audio_data_sets(data_dir, numcep, numcontext, 'test')
    
    # Create train, validation, and test DataSet's
    train = DataSet(inputs=train_inputs, outputs=train_outputs, seq_len=train_seq_len)
    validation = DataSet(inputs=validation_inputs, outputs=validation_outputs, seq_len=validation_seq_len)
    test = DataSet(inputs=test_inputs, outputs=test_outputs, seq_len=test_seq_len)
     
    # Return DataSets
    return DataSets(train=train, validation=validation, test=test)
    

def read_text_data_sets(data_dir, data_type):
    # TODO: Do not ignore data_type = ['train'|'validation'|'test']
    
    # Create file names
    text_filename = path.join(data_dir, 'LDC93S1.txt') 

    # Read text file and create list of sentence's words w/spaces replaced by ''
    with open(text_filename, 'rb') as f:
        for line in f.readlines():
            original = ' '.join(line.strip().lower().split(' ')[2:]).replace('.', '')
    
    return text_to_sparse_tuple([original])

def read_audio_data_sets(data_dir, numcep, numcontext, data_type):
    # TODO: Do not ignore data_type = ['train'|'validation'|'test']
     
    # Create file name
    audio_filename = path.join(data_dir, 'LDC93S1.wav') 

    # Return properly formatted data
    return audiofiles_to_audio_data_sets([audio_filename], numcep, numcontext)
