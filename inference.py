#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import numpy as np
import os
import pickle
import tensorflow as tf
import sys
import inspect
from util.audio import audiofile_to_input_vector
from util.text import ctc_label_dense_to_sparse, text_to_char_array, ndarray_to_text, wer
from util.spell import correction
import csv

tf.app.flags.DEFINE_integer('batch_size',     '1',     'batch_size for inference -- defaults to 1')
tf.app.flags.DEFINE_integer('n_hidden',        '',     'batch_size for inference')
tf.app.flags.DEFINE_string( 'dataset',         '',     'dataset to infer')
tf.app.flags.DEFINE_string( 'codebook_dir',    '',     'path to codebook directory')
tf.app.flags.DEFINE_integer('num_report',      10 ,    'number of sample results to report - defaults to 10 ')

FLAGS = tf.app.flags.FLAGS

class Codebook():

    def __init__(self, codebook_path, session=None):
        self.codebook = {}
        self.codebook_path = codebook_path
        self.filename = os.path.join(self.codebook_path, 'codebook_file')

    def load_codebook(self):
        with open(self.filename, 'rb') as fin:
            self.codebook = pickle.load(fin)

    def reconstruct_param(self):
        param = {}
        for p in self.codebook.keys():
            dic = {}
            for idx, i in enumerate(self.codebook[p][0]):
                dic[idx] = i
            # Reconstructing the cluster index from codebook dictionary
            param[p] = np.vectorize(dic.__getitem__)(self.codebook[p][1]).astype(np.float32)

        return param


class Data():
    def __init__(self, data_dir):

        self.dataset_dir = data_dir

    def get_data(self):
        csv_file_list = [os.path.join(self.dataset_dir, f) for f in os.listdir(self.dataset_dir) if f.endswith('.csv')]
        if len(csv_file_list) == 0:
            print('No csv files found')
            sys.exit()

        data = []
        for files in csv_file_list:
            with open(files) as f:
                reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_NONE)
                for row in reader:
                    data.append((row[0], row[1], row[2]))

        data.pop(0)

        return data


class Infer():

    def __init__(self, param, batch_size):
        self._param = param
        self.session = tf.Session()
        self.n_context = 9
        self.n_input = 26
        self.random_seed = tf.set_random_seed(4567)
        self.stddev = 0.046875
        self.relu_clip = 20

        self.n_hidden = FLAGS.n_hidden
        self.n_hidden_6 = 29
        self.no_dropout = [ 0.0 ] * 6
        self.n_cell_dim = self.n_hidden
        self.batch_size = batch_size

        # Parse and modify the input to compatible format
        self.target = tf.placeholder(tf.int32, [self.batch_size, None])
        self.target_len = tf.placeholder(tf.int32, [self.batch_size])
        self.input_tensor = tf.placeholder(tf.int32, [None, self.n_input + 2 * (self.n_input*self.n_context)])
        self.input_len = tf.placeholder(tf.int32, [self.batch_size])

    def get_embeddings(self, data):

        wav_file, _, transcript = data

        # Get the input from the wav_file
        input_vec = audiofile_to_input_vector(wav_file, self.n_input, self.n_context)

        # Convert the transcript to array
        target = text_to_char_array(transcript)

        return input_vec, np.asarray(target).reshape(self.batch_size, -1), transcript


    def retrieve_data(self, data):
        wav_file, _, transcript = data

        # Get the input from the wav_file
        input_vec = audiofile_to_input_vector(wav_file, self.n_input, self.n_context)

        # Convert the transcript to array
        target = text_to_char_array(transcript)

        return input_vec, len(input_vec), target, len(target), transcript


    def BiRNN(self, batch_x, seq_length, dropout):
        batch_x_shape = tf.shape(batch_x)

        # Reshaping `batch_x` to a tensor with shape `[n_steps*batch_size, n_input + 2*n_input*n_context]`.
        # This is done to prepare the batch for input into the first layer which expects a tensor of rank `2`.

        # Permute n_steps and batch_size
        batch_x = tf.transpose(batch_x, [1, 0, 2])
        # Reshape to prepare input for first layer
        batch_x = tf.reshape(batch_x, [-1, self.n_input + 2*self.n_input*self.n_context])

        # The next three blocks will pass `batch_x` through three hidden layers with
        # clipped RELU activation and dropout.

        # 1st layer
        h1 = tf.get_variable('h1', initializer=tf.constant(self._param['h1']))
        b1 = tf.get_variable('b1', initializer=tf.constant(self._param['b1']))
        layer_1 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(tf.cast(batch_x, tf.float32), h1), b1)), self.relu_clip)
        layer_1 = tf.nn.dropout(layer_1, (1.0 - dropout[0]))

        # 2nd layer
        h2 = tf.get_variable('h2', initializer=tf.constant(self._param['h2']))
        b2 = tf.get_variable('b2', initializer=tf.constant(self._param['b2']))
        layer_2 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(layer_1, h2), b2)), self.relu_clip)
        layer_2 = tf.nn.dropout(layer_2, (1.0 - dropout[1]))

        # 3rd layer
        b3 = tf.get_variable('b3', initializer=tf.constant(self._param['b3']))
        h3 = tf.get_variable('h3', initializer=tf.constant(self._param['h3']))
        layer_3 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(layer_2, h3), b3)), self.relu_clip)
        layer_3 = tf.nn.dropout(layer_3, (1.0 - dropout[2]))

        # Now we create the forward and backward LSTM units.
        # Both of which have inputs of length `n_cell_dim` and bias `1.0` for the forget gate of the LSTM.

        # Forward direction cell: (if else required for TF 1.0 and 1.1 compat)
        lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(self.n_cell_dim, forget_bias=1.0, state_is_tuple=True) \
                       if 'reuse' not in inspect.getargspec(tf.contrib.rnn.BasicLSTMCell.__init__).args else \
                       tf.contrib.rnn.BasicLSTMCell(self.n_cell_dim, forget_bias=1.0, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)
        lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(lstm_fw_cell,
                       input_keep_prob=1.0 - dropout[3],
                       output_keep_prob=1.0 - dropout[3],
                       seed=self.random_seed)
        # Backward direction cell: (if else required for TF 1.0 and 1.1 compat)
        lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(self.n_cell_dim, forget_bias=1.0, state_is_tuple=True) \
                       if 'reuse' not in inspect.getargspec(tf.contrib.rnn.BasicLSTMCell.__init__).args else \
                       tf.contrib.rnn.BasicLSTMCell(self.n_cell_dim, forget_bias=1.0, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)
        lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(lstm_bw_cell,
                                                     input_keep_prob=1.0 - dropout[4],
                                                     output_keep_prob=1.0 - dropout[4],
                                                     seed=self.random_seed)

        # `layer_3` is now reshaped into `[n_steps, batch_size, 2*n_cell_dim]`,
        # as the LSTM BRNN expects its input to be of shape `[max_time, batch_size, input_size]`.
        layer_3 = tf.reshape(layer_3, [-1, batch_x_shape[0], self.n_hidden*2])

        # Now we feed `layer_3` into the LSTM BRNN cell and obtain the LSTM BRNN output.
        outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell,
                                                                 cell_bw=lstm_bw_cell,
                                                                 inputs=layer_3,
                                                                 dtype=tf.float32,
                                                                 time_major=True,
                                                                 sequence_length=seq_length)

        # Reshape outputs from two tensors each of shape [n_steps, batch_size, n_cell_dim]
        # to a single tensor of shape [n_steps*batch_size, 2*n_cell_dim]
        outputs = tf.concat(outputs, 2)
        outputs = tf.reshape(outputs, [-1, 2*self.n_cell_dim])

        # Now we feed `outputs` to the fifth hidden layer with clipped RELU activation and dropout
        b5 = tf.get_variable('b5', initializer=tf.constant(self._param['b5']))
        h5 = tf.get_variable('h5', initializer = self._param['h5'])
        layer_5 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(outputs, h5), b5)), self.relu_clip)
        layer_5 = tf.nn.dropout(layer_5, (1.0 - dropout[5]))

        # Now we apply the weight matrix `h6` and bias `b6` to the output of `layer_5`
        # creating `n_classes` dimensional vectors, the logits.
        b6 = tf.get_variable('b6', initializer=tf.constant(self._param['b6']))
        h6 = tf.get_variable('h6', initializer= self._param['h6'])
        layer_6 = tf.add(tf.matmul(layer_5, h6), b6)

        # Finally we reshape layer_6 from a tensor of shape [n_steps*batch_size, n_hidden_6]
        # to the slightly more useful shape [n_steps, batch_size, n_hidden_6].
        # Note, that this differs from the input in that it is time-major.
        layer_6 = tf.reshape(layer_6, [-1, batch_x_shape[0], self.n_hidden_6])

        # Output shape: [n_steps, batch_size, n_hidden_6]
        return layer_6


    def do_inference(self):

        # input_tensor needs to be of 3 dimension [ batch_size, n_steps, n_input ]
        input_tensor = tf.expand_dims(self.input_tensor, axis=0)

        # Calculate the sequence_length, same as the input_length
        seq_length = self.input_len

        # Calculate the logits over the BiDirectional RNN modified as per inference part
        logits = self.BiRNN(input_tensor, seq_length, self.no_dropout)

        # Get the labels for calculating the ctc_loss
        sparse_label = ctc_label_dense_to_sparse(self.target, self.target_len, self.batch_size)

        # Calculate the ctc loss for logits and corresponding labels for a sequence length
        loss = tf.nn.ctc_loss(labels=sparse_label, inputs=logits, sequence_length=seq_length)

        # Average the loss
        avg_loss = tf.reduce_mean(loss)

        decoded, _ = tf.nn.ctc_beam_search_decoder(logits, seq_length, merge_repeated=False)
        pred = tf.convert_to_tensor([tf.sparse_tensor_to_dense(sparse_tensor) for sparse_tensor in decoded])
        dist = tf.edit_distance(tf.cast(decoded[0], tf.int32), sparse_label)
        acc = tf.reduce_mean(dist)
        return loss, avg_loss, dist, acc, pred, logits


    def init_lstm_param(self):

        # Assign the codebook generated parameters to LSTM cell weights
        assign_fw_w = tf.assign(self.session.graph.get_tensor_by_name('bidirectional_rnn/fw/basic_lstm_cell/weights:0'), self._param['bidirectional_rnn/fw/basic_lstm_cell/weights'])
        assign_bw_w = tf.assign(self.session.graph.get_tensor_by_name('bidirectional_rnn/bw/basic_lstm_cell/weights:0'), self._param['bidirectional_rnn/bw/basic_lstm_cell/weights'])
        assign_fw_b = tf.assign(self.session.graph.get_tensor_by_name('bidirectional_rnn/fw/basic_lstm_cell/biases:0'), self._param['bidirectional_rnn/fw/basic_lstm_cell/biases'].reshape([-1]))
        assign_bw_b = tf.assign(self.session.graph.get_tensor_by_name('bidirectional_rnn/bw/basic_lstm_cell/biases:0'), self._param['bidirectional_rnn/bw/basic_lstm_cell/biases'].reshape([-1]))

        return assign_fw_w, assign_bw_w, assign_fw_b, assign_bw_b

    def results(self, dataset):

        agg_loss = 0.0
        _acc = 0.0
        edit_dist = 0.0

        loss, avg_loss, dist, acc, pred, logits = self.do_inference()
        assign_fw_w, assign_bw_w, assign_fw_b, assign_bw_b = self.init_lstm_param()

        # Initialize the variables defined in the graph
        self.session.run(tf.global_variables_initializer())

        samples = []
        sum_wer = 0.0

        for idx,data in enumerate(dataset):
            # Retrive the input data
            input_x, target, transcript = self.get_embeddings(data)

            # LSTM param initialization
            self.session.run([assign_fw_w, assign_bw_w, assign_fw_b, assign_bw_b])

            seq_len = [len(input_x)]
            target_len = np.asarray([len(target[j]) for j in range(self.batch_size)])
            target = target.reshape(self.batch_size, -1)

            # Create the dictionary to feed the placeholders defined in the graph
            feed_dict = {self.target: target, self.target_len: target_len, self.input_tensor: input_x, self.input_len: seq_len}
            loss_, avg_loss_, dist_, acc_, prediction = self.session.run([loss, avg_loss, dist, acc, pred], feed_dict=feed_dict)
            agg_loss += avg_loss_
            edit_dist += dist_
            _acc += acc_
            text = ndarray_to_text(prediction[0][0])

            # Using Language Model
            text = correction(text)
            wer_value = wer(transcript, text)
            sum_wer += wer_value
            samples.append((wer_value, transcript, text, avg_loss_, dist_))
            print('Processed %d inputs out of %d ...' % (idx+1, len(dataset)))

        sorted(samples, key=lambda x: x[0], reverse=True)
        print('---------------------------------------------------------------')
        for i in range(len(dataset)%FLAGS.num_report):
            print(' WER: %f Loss: %f mean edit distance: %f' % (samples[i][0], samples[i][3], samples[i][4]))
            print(' - src: %s' % samples[i][1])
            print(' - res: %s\n' % samples[i][2])

        total = len(dataset)
        agg_loss /= total
        edit_dist /= total
        sum_wer /= total
        print('---------------------------------------------------------------')
        print('WER: %f Loss: %f Mean Edit distance: %f' %( sum_wer, agg_loss,  edit_dist))
        self.session.close()


def main(_):
    cb = Codebook(FLAGS.codebook_dir)

    # Load the codebook
    cb.load_codebook()

    # Reconstruct the parameters from codebook and cluster index matrix
    param = cb.reconstruct_param()

    test_dir = os.path.join('data', FLAGS.dataset)
    dat = Data(test_dir)
    data = dat.get_data()
    model = Infer(param, FLAGS.batch_size)
    model.results(data)



if __name__=='__main__':
    tf.app.run()

