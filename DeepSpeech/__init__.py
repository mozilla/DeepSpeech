import scipy.io.wavfile as wav
import numpy as np
import tensorflow as tf
import time

from python_speech_features import mfcc as MFCC
from utils import maybe_download as maybe_download

from tensorflow.python.ops import ctc_ops

import matplotlib.pyplot as plt
import csv

from natsort import natsorted

# Constants
SPACE_TOKEN = '<space>'
SPACE_INDEX = 0
FIRST_INDEX = ord('a') - 1  # 0 is reserved to space

class MfccContext:
    def __init__(self, _mfcc, _n_input, _n_steps, _n_context):
        self._mfcc      = _mfcc

        self._n_input   = _n_input
        self._n_steps   = _n_steps
        self._n_context = _n_context

    def _get_empty_mfcc(self):
        # Creating empty MFCC features to cope for time slices where we cannot get
        # sound (too close to the start/end for having a valid complete context)
        # TODO: We should replace this with MFCC features from silence, not 0
        empty_mfcc = np.array([])
        empty_mfcc.resize((self._n_input))
        ###print("empty_mfcc=", empty_mfcc, "shape=", empty_mfcc.shape)
        return empty_mfcc

    def contextualize(self):
        # Tranform in 3D array
        # >>> ar = np.array([[1, 2, 3],[4, 5, 6],[7, 8, 9],[10, 11, 12],[13, 14, 15]])
        # >>> ar
        # array([[ 1,  2,  3],
        #        [ 4,  5,  6],
        #        [ 7,  8,  9],
        #        [10, 11, 12],
        #        [13, 14, 15]])
        # >>> ar[np.newaxis, :]
        # array([[[ 1,  2,  3],
        #         [ 4,  5,  6],
        #         [ 7,  8,  9],
        #         [10, 11, 12],
        #         [13, 14, 15]]])
        orig_inputs = self._mfcc

        # For each time slice of the training set, we need to copy the context
        # this makes the 20 dimensions vector into a 220 dimensions
        # because of:
        #  - 20 dimensions for the current mfcc feature set
        #  - 5*20 dimensions for each of the past and future (x2) mfcc feature set
        # => so 20 + 2*5*20 = 220
        train_inputs = np.array([])
        train_inputs.resize((self._n_steps, self._n_input + 2*self._n_input*self._n_context))

        empty_mfcc = self._get_empty_mfcc()

        time_slices = range(train_inputs.shape[0])
        context_past_min   = time_slices[0]  + self._n_context
        context_future_max = time_slices[-1] - self._n_context
        for time_slice in time_slices:
            ### Reminder: array[start:stop:step]
            ### slices from indice |start| up to |stop| (not included), every |step|
            # Pick up to self._n_context time slices in the past, and complete with empty
            # mfcc features
            need_empty_past     = max(0, (context_past_min - time_slice))
            empty_source_past   = list(empty_mfcc for empty_slots in range(need_empty_past))
            data_source_past    = orig_inputs[max(0, time_slice - self._n_context):time_slice]
            assert(len(empty_source_past) + len(data_source_past) == self._n_context)

            # Pick up to self._n_context time slices in the future, and complete with empty
            # mfcc features
            need_empty_future   = max(0, (time_slice - context_future_max))
            empty_source_future = list(empty_mfcc for empty_slots in range(need_empty_future))
            data_source_future  = orig_inputs[time_slice + 1:time_slice + self._n_context + 1]
            assert(len(empty_source_future) + len(data_source_future) == self._n_context)

            if need_empty_past:
                past   = np.concatenate((empty_source_past, data_source_past))
            else:
                past   = data_source_past

            if need_empty_future:
                future = np.concatenate((data_source_future, empty_source_future))
            else:
                future = data_source_future

            past   = np.reshape(past, self._n_context*self._n_input)
            now    = orig_inputs[time_slice]
            future = np.reshape(future, self._n_context*self._n_input)

            train_inputs[time_slice] = np.concatenate((past, now, future))
            assert(len(train_inputs[time_slice]) == self._n_input + 2*self._n_input*self._n_context)

            ### print("train_inputs[", train_example, "][", time_slice, "]=", train_inputs[train_example][time_slice])

        # Normalize inputs
        train_inputs = (train_inputs - np.mean(train_inputs))/np.std(train_inputs)

        return train_inputs

class Input:
    def __init__(self, audio, target, _n_context):
        self._audio_filename  = maybe_download(audio[0], audio[1])
        self._target_filename = maybe_download(target[0], target[1])
        self._dimensions = {}
        self._dimensions['n_context'] = _n_context

    def _get_mfcc(self):
        fs, audio  = wav.read(self._audio_filename)
        mfcc_values = MFCC(audio, samplerate=fs)

        # Derivating base constants from samples
        ###batch_size = mfcc_values.shape[0]
        self._dimensions['n_steps'] = mfcc_values.shape[0]
        self._dimensions['n_input'] = mfcc_values.shape[1]

        return mfcc_values

    def _get_ready_mfcc_inputs(self):
        mfcc = self._get_mfcc()
        ctx  = MfccContext(mfcc, self._dimensions['n_input'], self._dimensions['n_steps'], self._dimensions['n_context'])
        return ctx.contextualize()

    def get_dimensions(self):
        return self._dimensions

    def prepare_net_inputs(self):
        return self._get_ready_mfcc_inputs()

    def prepare_net_targets(self):
        # Readings targets
        with open(self._target_filename, 'rb') as f:
            for line in f.readlines():
                if line[0] == ';':
                    continue

                # Get only the words between [a-z] and replace period for none
                original = ' '.join(line.strip().lower().split(' ')[2:]).replace('.', '')
                targets = original.replace(' ', '  ')
                targets = targets.split(' ')

        # Adding blank label
        targets = np.hstack([SPACE_TOKEN if xt == '' else list(xt) for xt in targets])

        # Transform char into index
        targets = np.asarray([SPACE_INDEX if xt == SPACE_TOKEN else ord(xt) - FIRST_INDEX
                              for xt in targets])

        return targets

class Network:
    def __init__(self, n_steps, n_input, n_context, n_character, batch_size, relu_clip):
        self.n_input = n_input
        self.n_context = n_context
        self.n_character = n_character
        self.n_steps = n_steps

        self.batch_size = batch_size
        self.relu_clip = relu_clip

        # Where `n_hidden_1` is the number of units in the first layer, `n_hidden_2` the number of units in the second, and  `n_hidden_5` the number in the fifth. We haven't forgotten about the third or sixth layer. We will define their unit count below.
        self.n_hidden_1 = self.n_input + 2*self.n_input*self.n_context # Note: This value was not specified in the original paper
        self.n_hidden_2 = self.n_input + 2*self.n_input*self.n_context # Note: This value was not specified in the original paper
        self.n_hidden_5 = self.n_input + 2*self.n_input*self.n_context # Note: This value was not specified in the original paper

        self.n_cell_dim = self.n_input + 2*self.n_input*self.n_context # TODO: Is this a reasonable value

        # The number of units in the third layer, which feeds in to the LSTM, is determined by `n_cell_dim` as follows
        self.n_hidden_3 = 2 * self.n_cell_dim

        # The number of units in the sixth layer is determined by `n_character` as follows
        self.n_hidden_6 = self.n_character

    # Graph Creation
    # Next we concern ourselves with graph creation.
    def BiRNN(self, _X, _weights, _biases):
        # Input shape: [batch_size, n_steps, self.n_input + 2*self.n_input*self.n_context]
        _X = tf.transpose(_X, [1, 0, 2])  # Permute n_steps and batch_size
        # Reshape to prepare input for first layer
        _X = tf.reshape(_X, [-1, self.n_input + 2*self.n_input*self.n_context]) # (n_steps*batch_size, self.n_input + 2*self.n_input*self.n_context)

        #Hidden layer with clipped RELU activation and dropout
        layer_1 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(_X, _weights['h1']), _biases['b1'])), self.relu_clip)
        layer_1 = tf.nn.dropout(layer_1, self.keep_prob)
        #Hidden layer with clipped RELU activation and dropout
        layer_2 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(layer_1, _weights['h2']), _biases['b2'])), self.relu_clip)
        layer_2 = tf.nn.dropout(layer_2, self.keep_prob)
        #Hidden layer with clipped RELU activation and dropout
        layer_3 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(layer_2, _weights['h3']), _biases['b3'])), self.relu_clip)
        layer_3 = tf.nn.dropout(layer_3, self.keep_prob)

        # Define lstm cells with tensorflow
        # Forward direction cell
        lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.n_cell_dim, forget_bias=1.0)
        # Backward direction cell
        lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.n_cell_dim, forget_bias=1.0)

        # Split data because rnn cell needs a list of inputs for the BRNN inner loop
        layer_3 = tf.split(0, self.n_steps, layer_3)

        # Get lstm cell output
        outputs, output_state_fw, output_state_bw = tf.nn.bidirectional_rnn(cell_fw=lstm_fw_cell,
                                                                            cell_bw=lstm_bw_cell,
                                                                            inputs=layer_3,
                                                                            dtype="float")

        # Reshape outputs from a list of n_steps tensors each of shape [batch_size, 2*n_cell_dim]
        # to a single tensor of shape [n_steps*batch_size, 2*n_cell_dim]
        outputs = tf.pack(outputs)
        outputs = tf.reshape(outputs, [-1, 2*self.n_cell_dim])

        #Hidden layer with clipped RELU activation and dropout
        layer_5 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(outputs, _weights['h5']), _biases['b5'])), self.relu_clip)
        layer_5 = tf.nn.dropout(layer_5, self.keep_prob)
        #Hidden layer without softmax function, ctc already takes care of softmax()
        layer_6 = tf.add(tf.matmul(layer_5, _weights['h6']), _biases['b6'])

        # Reshape layer_6 from a tensor of shape [n_steps*batch_size, n_hidden_6]
        # to a tensor of shape [batch_size, n_steps, n_hidden_6]
        layer_6 = tf.reshape(layer_6, [self.n_steps, self.batch_size, self.n_hidden_6])
        layer_6 = tf.transpose(layer_6, [1, 0, 2])  # Permute n_steps and batch_size

        # Return layer_6
        return layer_6

    def prepare(self, learning_rate, beta1, beta2, epsilon, use_warpctc=True):
        graph = tf.Graph()
        with graph.as_default():
            # First we create several place holders in our graph. The first two `x` and `y` are placeholders for our training data pairs.
            x = tf.placeholder("float", [None, self.n_steps, self.n_input + 2*self.n_input*self.n_context])
            y = tf.sparse_placeholder(tf.int32)

            # As `y` represents the text transcript of each element in a batch, it is of type \"string\" and has shape `[None, 1]` where the
            # `None` dimension corresponds to the number of elements in the batch.
            # The placeholder `x` is a place holder for the the speech spectrograms along with their prefix and postfix contexts for each element in a batch.
            # As it represents a spectrogram, its type is \"float\". The `None` dimension of its shape [None, n_steps, self.n_input + 2*self.n_input*self.n_context]
            # has the same meaning as the `None` dimension in the shape of `y`. The `n_steps` dimension of its shape indicates the number of time-slices in the sequence.
            # Finally, the `self.n_input + 2*self.n_input*self.n_context` dimension of its shape indicates the number of bins in Fourier transform `self.n_input`
            # along with the number of bins in the prefix-context `self.n_input*self.n_context` and postfix-contex `self.n_input*self.n_context`.

            # The next placeholders we introduce `istate_fw` and `istate_bw` correspond to the initial states and cells of the forward and backward LSTM networks.
            # As both of these are floats of dimension `n_cell_dim`, we define `istate_fw` and `istate_bw` as follows
            istate_fw = (tf.placeholder("float", [None, self.n_cell_dim]), tf.placeholder("float", [None, self.n_cell_dim]))
            istate_bw = (tf.placeholder("float", [None, self.n_cell_dim]), tf.placeholder("float", [None, self.n_cell_dim]))
            self.keep_prob = tf.placeholder(tf.float32)

            # Store layers weight & bias
            # TODO: Is random_normal the best distribution to draw from?
            weights = {
                'h1': tf.Variable(tf.random_normal([self.n_input + 2*self.n_input*self.n_context, self.n_hidden_1])),
                'h2': tf.Variable(tf.random_normal([self.n_hidden_1, self.n_hidden_2])),
                'h3': tf.Variable(tf.random_normal([self.n_hidden_2, self.n_hidden_3])),
                'h5': tf.Variable(tf.random_normal([(2 * self.n_cell_dim), self.n_hidden_5])),
                'h6': tf.Variable(tf.random_normal([self.n_hidden_5, self.n_hidden_6]))
            }
            biases = {
                'b1': tf.Variable(tf.random_normal([self.n_hidden_1])),
                'b2': tf.Variable(tf.random_normal([self.n_hidden_2])),
                'b3': tf.Variable(tf.random_normal([self.n_hidden_3])),
                'b5': tf.Variable(tf.random_normal([self.n_hidden_5])),
                'b6': tf.Variable(tf.random_normal([self.n_hidden_6]))
            }

            # Here we use sparse_placeholder that will generate a
            # SparseTensor required by ctc_loss op.
            targets = tf.sparse_placeholder(tf.int32)

            # 1d array of size [batch_size]
            seq_len = tf.placeholder(tf.int32, [None])

            # The second output is the last state and we will no use that
            ## outputs, _ = tf.nn.dynamic_rnn(cell, inputs, seq_len, dtype=tf.float32)
            layer_6 = self.BiRNN(x, weights, biases)

            # Convert to time major for CTC
            layer_6 = tf.transpose(layer_6, (1, 0, 2))

            if use_warpctc == False:
                print("Using CTC loss")
                loss = ctc_ops.ctc_loss(layer_6, targets, seq_len)
            else:
                print("Using WarpCTC loss")
                loss = tf.contrib.warpctc.warp_ctc_loss(layer_6, targets, seq_len)

            cost = tf.reduce_mean(loss)

            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, beta2=beta2, epsilon=epsilon).minimize(cost)

            # Option 2: tf.contrib.ctc.ctc_beam_search_decoder
            # (it's slower but you'll get better results)
            decoded, log_prob = ctc_ops.ctc_greedy_decoder(layer_6, seq_len)

            # Accuracy: label error rate
            acc = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32),
                                                  targets))

            return graph, x, y, targets, seq_len, self.keep_prob, loss, cost, optimizer, acc, decoded

class Trainer:
    def __init__(self, graph, pX, pTargets, pSeqLen, pKeepProb, fCost, fOptimizer, fAccuracy, pDecoded):
        self._graph     = graph
        self._x         = pX
        self._targets   = pTargets
        self._seq_len   = pSeqLen
        self._keep_prob = pKeepProb
        self._cost      = fCost
        self._optimizer = fOptimizer
        self._accuracy  = fAccuracy
        self._decoded   = pDecoded

    def train(self, epochs, train_inputs, val_inputs, train_targets, val_targets, train_seq_len, val_seq_len, train_keep_prob, val_keep_prob, step_cb):
        with tf.Session(graph=self._graph) as session:
            tf.initialize_all_variables().run()

            for curr_epoch in xrange(epochs):
                train_cost = train_ler = 0
                start = time.time()

                feed = {self._x:         train_inputs,
                        self._targets:   train_targets,
                        self._seq_len:   train_seq_len,
                        self._keep_prob: train_keep_prob}

                batch_cost, _ = session.run([self._cost, self._optimizer], feed)
                train_cost   += batch_cost
                train_ler    += session.run(self._accuracy, feed_dict=feed)

                val_feed = {self._x:         val_inputs,
                            self._targets:   val_targets,
                            self._seq_len:   val_seq_len,
                            self._keep_prob: val_keep_prob}

                val_cost, val_ler = session.run([self._cost, self._accuracy], feed_dict=val_feed)

                if step_cb:
                    step_cb(curr_epoch, epochs, train_cost, train_ler, val_cost, val_ler, time.time() - start)

            # Decoding
            return Decoder(session.run(self._decoded[0], feed_dict=feed)).decode()

class Decoder:
    def __init__(self, value):
        self._str = value

    def decode(self):
        str_decoded = ''.join([chr(xt) for xt in np.asarray(self._str[1]) + FIRST_INDEX])
        # Replacing blank label to none
        str_decoded = str_decoded.replace(chr(ord('z') + 1), '')
        # Replacing space label to space
        str_decoded = str_decoded.replace(chr(ord('a') - 1), ' ')

        return str_decoded

class Plotter:

    def __init__(self, in_fname, out_fname, plot_type):
        self._png = out_fname
        self._csv = in_fname

        self.load_csv()
        self.plot(plot_type)

    def load_csv(self):
        """ the csv file we read is made:
           'run', 'epoch', 'x', 'cost', 'valerr', 'time'
        """
        self._csv_content = {}
        maxRun  = 0
        maxIter = 0
        csv_file = csv.DictReader(self._csv, delimiter=',', quotechar='"')
        for line in csv_file:
            r = int(line['run'])
            try:
                self._csv_content[r].append(line)
            except KeyError as e:
                self._csv_content[r] = [ line ]
            maxRun  = max(maxRun, int(line['run']))
            maxIter = max(maxIter, int(line['epoch']))
        self._csv.close()

        self._X = xrange(maxIter + 1)
        self._C = np.ndarray(shape=(maxRun + 1, maxIter + 1), dtype=float)
        self._E = np.ndarray(shape=(maxRun + 1, maxIter + 1), dtype=float)
        self._T = np.ndarray(shape=(maxRun + 1, maxIter + 1), dtype=float)

        for run in xrange(maxRun+1):
            c = self._csv_content[run]
            self._C[run] = map(lambda x: x['cost'], c)
            self._E[run] = map(lambda x: x['valerr'], c)
            self._T[run] = map(lambda x: x['time'], c)

        self._meanC = np.mean(self._C, axis=0)
        self._ecC   = np.std(self._C, axis=0)

        self._meanE = np.mean(self._E, axis=0)
        self._ecE   = np.std(self._E, axis=0)

        self._meanT = np.mean(self._T, axis=0)
        self._ecT   = np.std(self._T, axis=0)

    def plot(self, type="loss"):
        if type == "loss":
            self.plot_loss()
        elif type == "valerr":
            self.plot_val_err()
	elif type == "time":
	    self.plot_time()
        ##else:
        ##    print("No plotting supported:", type)

    def plot_loss(self):
        fig, axis1 = plt.subplots()
        axis1.errorbar(self._X, self._meanC, yerr=self._ecC, color="red", label="loss")
        axis1.legend(loc="upper left", frameon=False)
        axis1.set_autoscaley_on(False)
        axis1.set_ylim([1e0, 1.1e3])

        axis2 = axis1.twinx()
        axis2.errorbar(self._X, self._meanT, yerr=self._ecT, color="blue", label="time")
        axis2.legend(loc="upper right", frameon=False)
        axis2.set_autoscaley_on(False)
        axis2.set_ylim([0, 1e1])

        axis1.set_xlabel("Epochs")
        axis1.set_yscale('log')

        fig.set_size_inches(24, 18)
        plt.title("Loss evolution")
        plt.savefig(self._png, dpi=100)

    def plot_val_err(self):
        fig, axis1 = plt.subplots()
        axis1.errorbar(self._X, self._meanE, yerr=self._ecE, color="red", label="valid. error")
        axis1.legend(loc="upper left", frameon=False)
        axis1.set_autoscaley_on(False)
        axis1.set_ylim([1e-5, 1e1])

        axis2 = axis1.twinx()
        axis2.errorbar(self._X, self._meanT, yerr=self._ecT, color="blue", label="time")
        axis2.legend(loc="upper right", frameon=False)
        axis2.set_autoscaley_on(False)
        axis2.set_ylim([0, 1e1])

        axis1.set_xlabel("Epochs")
        axis1.set_yscale('log')

        fig.set_size_inches(24, 18)
        plt.title("Validation error evolution")
        plt.savefig(self._png, dpi=100)

    def plot_time(self):
        fig, axis1 = plt.subplots()
        axis1.errorbar(self._X, self._meanT, yerr=self._ecT, color="blue", label="time")
        axis1.legend(loc="upper right", frameon=False)

        fig.set_size_inches(24, 18)
        plt.title("Execution time")
        plt.savefig(self._png, dpi=100)

class MultiPlotter(Plotter):

    def __init__(self, files, plot, type, title, xlbl, xtics, ylbl):
	self._files = natsorted(files, key=lambda y: y.name.lower())
        self._png   = plot
	self._title = title
	self._xlbl  = xlbl
	self._xtics = xtics
	self._ylbl  = ylbl

	self.load_csv()
	Plotter.plot(self, type)

    def load_csv(self):
        self._plotters = []
	for file in self._files:
	    self._plotters.append(Plotter(file, None, None))

        self._X = xrange(len(self._plotters))

	self._meanC = [ np.mean(X._C) for X in self._plotters ]
        self._ecC   = [ np.std(X._C) for X in  self._plotters ]

	self._meanE = [ np.mean(X._E) for X in self._plotters ]
        self._ecE   = [ np.std(X._E) for X in  self._plotters ]

	self._meanT = [ np.mean(X._T) for X in self._plotters ]
        self._ecT   = [ np.std(X._T) for X in  self._plotters ]

    def plot_time(self):
        fig, axis1 = plt.subplots()
        axis1.errorbar(self._X, self._meanT, yerr=self._ecT, color="blue", label="time")
        axis1.legend(loc="upper right", frameon=False)

	if self._xlbl:
            axis1.set_xlabel(self._xlbl)

        if self._xtics:
            axis1.set_xticks(self._xtics)

	if self._ylbl:
            axis1.set_ylabel(self._ylbl)

        fig.set_size_inches(24, 18)
	if self._title is None:
            plt.title("Execution time")
	else:
            plt.title(self._title)

        plt.savefig(self._png, dpi=100)
