
# coding: utf-8

# # Introduction

# In this notebook we will reproduce the results of [Deep Speech: Scaling up end-to-end speech recognition](http://arxiv.org/abs/1412.5567). The core of the system is a bidirectional recurrent neural network (BRNN) trained to ingest speech spectrograms and generate English text transcriptions.
# 
#  Let a single utterance $x$ and label $y$ be sampled from a training set $S = \{(x^{(1)}, y^{(1)}), (x^{(2)}, y^{(2)}), . . .\}$. Each utterance, $x^{(i)}$ is a time-series of length $T^{(i)}$ where every time-slice is a vector of audio features, $x^{(i)}_t$ where $t=1,\ldots,T^{(i)}$. We use MFCC as our features; so $x^{(i)}_{t,p}$ denotes the $p$-th MFCC feature in the audio frame at time $t$. The goal of our BRNN is to convert an input sequence $x$ into a sequence of character probabilities for the transcription $y$, with $\hat{y}_t =\mathbb{P}(c_t \mid x)$, where $c_t \in \{a,b,c, . . . , z, space, apostrophe, blank\}$. (The significance of $blank$ will be explained below.)
# 
# Our BRNN model is composed of $5$ layers of hidden units. For an input $x$, the hidden units at layer $l$ are denoted $h^{(l)}$ with the convention that $h^{(0)}$ is the input. The first three layers are not recurrent. For the first layer, at each time $t$, the output depends on the MFCC frame $x_t$ along with a context of $C$ frames on each side. (We typically use $C \in \{5, 7, 9\}$ for our experiments.) The remaining non-recurrent layers operate on independent data for each time step. Thus, for each time $t$, the first $3$ layers are computed by:
# 
# $$h^{(l)}_t = g(W^{(l)} h^{(l-1)}_t + b^{(l)})$$
# 
# where $g(z) = \min\{\max\{0, z\}, 20\}$ is a clipped rectified-linear (ReLu) activation function and $W^{(l)}$, $b^{(l)}$ are the weight matrix and bias parameters for layer $l$. The fourth layer is a bidirectional recurrent layer[[1](http://www.di.ufpe.br/~fnj/RNA/bibliografia/BRNN.pdf)]. This layer includes two sets of hidden units: a set with forward recurrence, $h^{(f)}$, and a set with backward recurrence $h^{(b)}$:
# 
# $$h^{(f)}_t = g(W^{(4)} h^{(3)}_t + W^{(f)}_r h^{(f)}_{t-1} + b^{(4)})$$
# $$h^{(b)}_t = g(W^{(4)} h^{(3)}_t + W^{(b)}_r h^{(b)}_{t+1} + b^{(4)})$$
# 
# Note that $h^{(f)}$ must be computed sequentially from $t = 1$ to $t = T^{(i)}$ for the $i$-th utterance, while
# the units $h^{(b)}$ must be computed sequentially in reverse from $t = T^{(i)}$ to $t = 1$.
# 
# The fifth (non-recurrent) layer takes both the forward and backward units as inputs
# 
# $$h^{(5)} = g(W^{(5)} h^{(4)} + b^{(5)})$$
# 
# where $h^{(4)} = h^{(f)} + h^{(b)}$. The output layer are standard logits that correspond to the predicted character probabilities for each time slice $t$ and character $k$ in the alphabet:
# 
# $$h^{(6)}_{t,k} = \hat{y}_{t,k} = (W^{(6)} h^{(5)}_t)_k + b^{(6)}_k$$
# 
# Here $b^{(6)}_k$ denotes the $k$-th bias and $(W^{(6)} h^{(5)}_t)_k$ the $k$-th element of the matrix product.
# 
# Once we have computed a prediction for $\hat{y}_{t,k}$, we compute the CTC loss[[2]](http://www.cs.toronto.edu/~graves/preprint.pdf) $\cal{L}(\hat{y}, y)$ to measure the error in prediction. During training, we can evaluate the gradient $\nabla \cal{L}(\hat{y}, y)$ with respect to the network outputs given the ground-truth character sequence $y$. From this point, computing the gradient with respect to all of the model parameters may be done via back-propagation through the rest of the network. We use the Adam method for training[[3](http://arxiv.org/abs/1412.6980)].
# 
# The complete BRNN model is illustrated in the figure below.
# 
# ![DeepSpeech BRNN](images/rnn_fig-624x548.png)
# 
# 

# # Preliminaries

# ## Imports

# Here we first import all of the packages we require to implement the DeepSpeech BRNN.

# In[ ]:

import os
import sys
import time
import json
import shutil
import datetime
import tempfile
import subprocess
import numpy as np
from math import ceil
from xdg import BaseDirectory as xdg
import tensorflow as tf
from util.log import merge_logs
from util.gpu import get_available_gpus
from util.shared_lib import check_cupti
from util.text import sparse_tensor_value_to_texts, wer
from tensorflow.python.ops import ctc_ops
from tensorflow.contrib.session_bundle import exporter
from collections import OrderedDict

ds_importer = os.environ.get('ds_importer', 'ldc93s1')
ds_dataset_path = os.environ.get('ds_dataset_path', os.path.join('./data', ds_importer))

import importlib
ds_importer_module = importlib.import_module('util.importers.%s' % ds_importer)

from util.website import maybe_publish

do_fulltrace = bool(len(os.environ.get('ds_do_fulltrace', '')))

if do_fulltrace:
    check_cupti()


# ## Global Constants

# Next we introduce several constants used in the algorithm below.  In particular, we define
# * `learning_rate` - The learning rate we will employ in Adam optimizer[[3]](http://arxiv.org/abs/1412.6980)
# * `epochs` - The number of iterations (epochs) we will train for
# * `train_batch_size` - The number of elements in a training batch
# * `dev_batch_size` - The number of elements in a dev batch
# * `test_batch_size` - The number of elements in a test batch
# * `display_step` - The number of epochs we cycle through before displaying progress
# * `report_count` - The number of phrases to print out during a WER report
# * `checkpoint_step` - The number of epochs we cycle through before checkpointing the model
# * `limit_train` - The maximum amount of samples taken from (the beginning of) the train set - 0 meaning no limit
# * `limit_dev` - The maximum amount of samples taken from (the beginning of) the validation set - 0 meaning no limit
# * `limit_test` - The maximum amount of samples taken from (the beginning of) the test set - 0 meaning no limit
# * `checkpoint_dir` - The directory in which checkpoints are stored
# * `restore_checkpoint` - Whether to resume from checkpoints when training
# * `export_dir` - The directory in which exported models are stored
# * `export_version` - The version number of the exported model
# * `remove_export` - Whether to remove old exported models
# * `default_stddev` - The default standard deviation to use when initialising weights and biases
# * `log_device_placement` - Whether to log device placement of the operators to the console
# * `random_seed` - The default random seed that is used to initialize variables. Ensures reproducibility.
# * `[bh][12356]_stddev` - Individual standard deviations to use when initialising particular weights and biases
# * `log_variables` - Whether to log gradients and variables summaries to TensorBoard during training. This incurs a performance hit, so should generally only be enabled for debugging.

# In[ ]:

learning_rate = float(os.environ.get('ds_learning_rate', 0.001))
beta1 = float(os.environ.get('ds_beta1', 0.9))                   # TODO: Determine a reasonable value for this
beta2 = float(os.environ.get('ds_beta2', 0.999))                 # TODO: Determine a reasonable value for this
epsilon = float(os.environ.get('ds_epsilon', 1e-8))              # TODO: Determine a reasonable value for this
epochs = int(os.environ.get('ds_epochs', 75))
train_batch_size = int(os.environ.get('ds_train_batch_size', 1))
dev_batch_size = int(os.environ.get('ds_dev_batch_size', 1))
test_batch_size = int(os.environ.get('ds_test_batch_size', 1))
display_step = int(os.environ.get('ds_display_step', 1))
report_count = int(os.environ.get('ds_report_count', 10))
validation_step = int(os.environ.get('ds_validation_step', 0))
checkpoint_step = int(os.environ.get('ds_checkpoint_step', 5))
limit_train = int(os.environ.get('ds_limit_train', 0))
limit_dev   = int(os.environ.get('ds_limit_dev',   0))
limit_test  = int(os.environ.get('ds_limit_test',  0))
checkpoint_dir = os.environ.get('ds_checkpoint_dir', xdg.save_data_path('deepspeech'))
restore_checkpoint = bool(int(os.environ.get('ds_restore_checkpoint', 0)))
export_dir = os.environ.get('ds_export_dir', None)
export_version = 1
remove_export = bool(int(os.environ.get('ds_remove_export', 0)))
use_warpctc = bool(len(os.environ.get('ds_use_warpctc', '')))
default_stddev = float(os.environ.get('ds_default_stddev', 0.046875))
log_device_placement = bool(int(os.environ.get('ds_log_device_placement', 0)))
random_seed = int(os.environ.get('ds_random_seed', 4567)) # To be adjusted in case of bad luck
for var in ['b1', 'h1', 'b2', 'h2', 'b3', 'h3', 'b5', 'h5', 'b6', 'h6']:
    locals()['%s_stddev' % var] = float(os.environ.get('ds_%s_stddev' % var, default_stddev))
log_variables = bool(len(os.environ.get('ds_log_variables', '')))


# Note that we use the Adam optimizer[[3]](http://arxiv.org/abs/1412.6980) instead of Nesterov’s Accelerated Gradient [[4]](http://www.cs.utoronto.ca/~ilya/pubs/2013/1051_2.pdf) used in the original DeepSpeech paper, as, at the time of writing, TensorFlow does not have an implementation of Nesterov’s Accelerated Gradient [[4]](http://www.cs.utoronto.ca/~ilya/pubs/2013/1051_2.pdf).
# 
# As we will also employ dropout on the feedforward layers of the network, we need to define a parameter `dropout_rate` that keeps track of the dropout rate for these layers

# In[ ]:

dropout_rate = float(os.environ.get('ds_dropout_rate', 0.05))  # TODO: Validate this is a reasonable value


# One more constant required of the non-recurrant layers is the clipping value of the ReLU. We capture that in the value of the variable `relu_clip`

# In[ ]:

relu_clip = int(os.environ.get('ds_relu_clip', 20)) # TODO: Validate this is a reasonable value


# Let's also introduce a standard session configuration that'll be used for all new sessions.

# In[ ]:

session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=log_device_placement)


# ## Geometric Constants

# Now we will introduce several constants related to the geometry of the network.
# 
# The network views each speech sample as a sequence of time-slices $x^{(i)}_t$ of length $T^{(i)}$. As the speech samples vary in length, we know that $T^{(i)}$ need not equal $T^{(j)}$ for $i \ne j$. For each batch, BRNN in TensorFlow needs to know `n_steps` which is the maximum $T^{(i)}$ for the batch.

# Each of the at maximum `n_steps` vectors is a vector of MFCC features of a time-slice of the speech sample. We will make the number of MFCC features dependent upon the sample rate of the data set. Generically, if the sample rate is 8kHz we use 13 features. If the sample rate is 16kHz we use 26 features... We capture the dimension of these vectors, equivalently the number of MFCC features, in the variable `n_input`

# In[ ]:

n_input = 26 # TODO: Determine this programatically from the sample rate


# As previously mentioned, the BRNN is not simply fed the MFCC features of a given time-slice. It is fed, in addition, a context of $C \in \{5, 7, 9\}$ frames on either side of the frame in question. The number of frames in this context is captured in the variable `n_context`

# In[ ]:

n_context = 9 # TODO: Determine the optimal value using a validation data set


# Next we will introduce constants that specify the geometry of some of the non-recurrent layers of the network. We do this by simply specifying the number of units in each of the layers

# In[ ]:

n_hidden_1 = 2048
n_hidden_2 = 2048
n_hidden_5 = 2048


# where `n_hidden_1` is the number of units in the first layer, `n_hidden_2` the number of units in the second, and  `n_hidden_5` the number in the fifth. We haven't forgotten about the third or sixth layer. We will define their unit count below.

# A LSTM BRNN consists of a pair of LSTM RNN's. One LSTM RNN that works "forward in time"

# <img src="images/LSTM3-chain.png" alt="LSTM" width="800">

# and a second LSTM RNN that works "backwards in time"

# <img src="images/LSTM3-chain.png" alt="LSTM" width="800">

# The dimension of the cell state, the upper line connecting subsequent LSTM units, is independent of the input dimension and the same for both the forward and backward LSTM RNN.
# 
# Hence, we are free to choose the dimension of this cell state independent of the input dimension. We capture the cell state dimension in the variable `n_cell_dim`.

# In[ ]:

n_cell_dim = 2048


# The number of units in the third layer, which feeds in to the LSTM, is determined by `n_cell_dim` as follows

# In[ ]:

n_hidden_3 = 2 * n_cell_dim


# Next, we introduce an additional variable `n_character` which holds the number of characters in the target language plus one, for the $blamk$. For English it is the cardinality of the set $\{a,b,c, . . . , z, space, apostrophe, blank\}$ we referred to earlier.

# In[ ]:

n_character = 29 # TODO: Determine if this should be extended with other punctuation


# The number of units in the sixth layer is determined by `n_character` as follows 

# In[ ]:

n_hidden_6 = n_character


# # Graph Creation

# Next we concern ourselves with graph creation.
# 
# However, before we do so we must introduce a utility function `variable_on_cpu()` used to create a variable in CPU memory.

# In[ ]:

def variable_on_cpu(name, shape, initializer):
    # Use the /cpu:0 device for scoped operations
    with tf.device('/cpu:0'):
        # Create or get apropos variable
        var = tf.get_variable(name=name, shape=shape, initializer=initializer)
    return var


# That done, we will define the learned variables, the weights and biases, within the method `BiRNN()` which also constructs the neural network. The  variables named `hn`, where `n` is an integer, hold the learned weight variables. The variables named `bn`, where `n` is an integer, hold the learned bias variables.
# 
# In particular, the first variable `h1` holds the learned weight matrix that converts an input vector of dimension `n_input + 2*n_input*n_context`  to a vector of dimension `n_hidden_1`. Similarly, the second variable `h2` holds the weight matrix converting an input vector of dimension `n_hidden_1` to one of dimension `n_hidden_2`. The variables `h3`, `h5`, and `h6` are similar. Likewise, the biases, `b1`, `b2`..., hold the biases for the various layers.
# 
# That said let us introduce the method `BiRNN()` that takes a batch of data `batch_x` and performs inference upon it.

# In[ ]:

def BiRNN(batch_x, seq_length, dropout):
    # Input shape: [batch_size, n_steps, n_input + 2*n_input*n_context]
    batch_x_shape = tf.shape(batch_x)
    # Permute n_steps and batch_size
    batch_x = tf.transpose(batch_x, [1, 0, 2])
    # Reshape to prepare input for first layer
    batch_x = tf.reshape(batch_x, [-1, n_input + 2*n_input*n_context]) # (n_steps*batch_size, n_input + 2*n_input*n_context)
    
    #Hidden layer with clipped RELU activation and dropout
    b1 = variable_on_cpu('b1', [n_hidden_1], tf.random_normal_initializer(stddev=b1_stddev))
    h1 = variable_on_cpu('h1', [n_input + 2*n_input*n_context, n_hidden_1], tf.random_normal_initializer(stddev=h1_stddev))
    layer_1 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(batch_x, h1), b1)), relu_clip)
    layer_1 = tf.nn.dropout(layer_1, (1.0 - dropout))
    #Hidden layer with clipped RELU activation and dropout
    b2 = variable_on_cpu('b2', [n_hidden_2], tf.random_normal_initializer(stddev=b2_stddev))
    h2 = variable_on_cpu('h2', [n_hidden_1, n_hidden_2], tf.random_normal_initializer(stddev=h2_stddev))
    layer_2 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(layer_1, h2), b2)), relu_clip)
    layer_2 = tf.nn.dropout(layer_2, (1.0 - dropout))
    #Hidden layer with clipped RELU activation and dropout
    b3 = variable_on_cpu('b3', [n_hidden_3], tf.random_normal_initializer(stddev=b3_stddev))
    h3 = variable_on_cpu('h3', [n_hidden_2, n_hidden_3], tf.random_normal_initializer(stddev=h3_stddev))
    layer_3 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(layer_2, h3), b3)), relu_clip)
    layer_3 = tf.nn.dropout(layer_3, (1.0 - dropout))
    
    # Define lstm cells with tensorflow
    # Forward direction cell
    lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_cell_dim, forget_bias=1.0, state_is_tuple=True)
    # Backward direction cell
    lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_cell_dim, forget_bias=1.0, state_is_tuple=True)
    
    # Reshape data because rnn cell expects shape [max_time, batch_size, input_size]
    layer_3 = tf.reshape(layer_3, [-1, batch_x_shape[0], n_hidden_3])

    # Get lstm cell output
    outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell,
                                                             cell_bw=lstm_bw_cell,
                                                             inputs=layer_3,
                                                             dtype=tf.float32,
                                                             time_major=True,
                                                             sequence_length=seq_length)
    
    # Reshape outputs from two tensors each of shape [n_steps, batch_size, n_cell_dim]
    # to a single tensor of shape [n_steps*batch_size, 2*n_cell_dim]
    outputs = tf.concat(2, outputs)
    outputs = tf.reshape(outputs, [-1, 2*n_cell_dim])
    
    #Hidden layer with clipped RELU activation and dropout
    b5 = variable_on_cpu('b5', [n_hidden_5], tf.random_normal_initializer(stddev=b5_stddev))
    h5 = variable_on_cpu('h5', [(2 * n_cell_dim), n_hidden_5], tf.random_normal_initializer(stddev=h5_stddev))
    layer_5 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(outputs, h5), b5)), relu_clip)
    layer_5 = tf.nn.dropout(layer_5, (1.0 - dropout))
    #Hidden layer of logits
    b6 = variable_on_cpu('b6', [n_hidden_6], tf.random_normal_initializer(stddev=b6_stddev))
    h6 = variable_on_cpu('h6', [n_hidden_5, n_hidden_6], tf.random_normal_initializer(stddev=h6_stddev))
    layer_6 = tf.add(tf.matmul(layer_5, h6), b6)
    
    # Reshape layer_6 from a tensor of shape [n_steps*batch_size, n_hidden_6]
    # to a tensor of shape [n_steps, batch_size, n_hidden_6]
    layer_6 = tf.reshape(layer_6, [-1, batch_x_shape[0], n_hidden_6])
    
    # Return layer_6
    # Output shape: [n_steps, batch_size, n_hidden_6]
    return layer_6


# The first few lines of the function `BiRNN`
# ```python
# def BiRNN(batch_x, seq_length, dropout):
#     # Input shape: [batch_size, n_steps, n_input + 2*n_input*n_context]
#     batch_x_shape = tf.shape(batch_x)
#     # Permute n_steps and batch_size
#     batch_x = tf.transpose(batch_x, [1, 0, 2])
#     # Reshape to prepare input for first layer
#     batch_x = tf.reshape(batch_x, [-1, n_input + 2*n_input*n_context])
#     ...
# ```
# reshape `batch_x` which has shape `[batch_size, n_steps, n_input + 2*n_input*n_context]` initially, to a tensor with shape `[n_steps*batch_size, n_input + 2*n_input*n_context]`. This is done to prepare the batch for input into the first layer which expects a tensor of rank `2`.
# 
# The next few lines of  `BiRNN`
# ```python
#     #Hidden layer with clipped RELU activation and dropout
#     b1 = variable_on_cpu('b1', [n_hidden_1], tf.random_normal_initializer())
#     h1 = variable_on_cpu('h1', [n_input + 2*n_input*n_context, n_hidden_1], tf.random_normal_initializer())
#     layer_1 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(batch_x, h1), b1)), relu_clip)
#     layer_1 = tf.nn.dropout(layer_1, (1.0 - dropout))
#     ...
# ```
# pass `batch_x` through the first layer of the non-recurrent neural network, then applies dropout to the result.
# 
# The next few lines do the same thing, but for the second and third layers
# ```python
#     #Hidden layer with clipped RELU activation and dropout
#     b2 = variable_on_cpu('b2', [n_hidden_2], tf.random_normal_initializer())
#     h2 = variable_on_cpu('h2', [n_hidden_1, n_hidden_2], tf.random_normal_initializer())
#     layer_2 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(layer_1, h2), b2)), relu_clip)   
#     layer_2 = tf.nn.dropout(layer_2, (1.0 - dropout))
#     #Hidden layer with clipped RELU activation and dropout
#     b3 = variable_on_cpu('b3', [n_hidden_3], tf.random_normal_initializer())
#     h3 = variable_on_cpu('h3', [n_hidden_2, n_hidden_3], tf.random_normal_initializer())
#     layer_3 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(layer_2, h3), b3)), relu_clip)
#     layer_3 = tf.nn.dropout(layer_3, (1.0 - dropout))
# ```
# 
# Next we create the forward and backward LSTM units
# ```python
#     # Define lstm cells with tensorflow
#     # Forward direction cell
#     lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_cell_dim, forget_bias=1.0)
#     # Backward direction cell
#     lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_cell_dim, forget_bias=1.0)
# ```
# both of which have inputs of length `n_cell_dim` and bias `1.0` for the forget gate of the LSTM.
# 
# The next line of the funtion `BiRNN` does a bit more data preparation.
# ```python
#     # Reshape data because rnn cell expects shape [max_time, batch_size, input_size]
#     layer_3 = tf.reshape(layer_3, [-1, batch_x_shape[0], n_hidden_3])
# ```
# It reshapes `layer_3` in to `[n_steps, batch_size, 2*n_cell_dim]` as the LSTM BRNN expects its input to be of shape `[max_time, batch_size, input_size]`.
# 
# The next line of `BiRNN`
# ```python
#     # Get lstm cell output
#     outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell,
#                                                              cell_bw=lstm_bw_cell,
#                                                              inputs=layer_3,
#                                                              dtype=tf.float32,
#                                                              time_major=True,
#                                                              sequence_length=seq_length)
# ```
# feeds `layer_3` to the LSTM BRNN cell and obtains the LSTM BRNN output.
# 
# The next lines convert `outputs` from two rank two tensors into a single rank two tensor in preparation for passing it to the next neural network layer  
# ```python
#     # Reshape outputs from two tensors each of shape [n_steps, batch_size, n_cell_dim]
#     # to a single tensor of shape [n_steps*batch_size, 2*n_cell_dim]
#     outputs = tf.concat(2, outputs)
#     outputs = tf.reshape(outputs, [-1, 2*n_cell_dim])
# ```
# 
# The next couple of lines feed `outputs` to the fifth hidden layer
# ```python
#     #Hidden layer with clipped RELU activation and dropout
#     b5 = variable_on_cpu('b5', [n_hidden_5], tf.random_normal_initializer())
#     h5 = variable_on_cpu('h5', [(2 * n_cell_dim), n_hidden_5], tf.random_normal_initializer())
#     layer_5 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(outputs, h5), b5)), relu_clip)
#     layer_5 = tf.nn.dropout(layer_5, (1.0 - dropout))
# ```
# 
# The next line of `BiRNN`
# ```python
#     #Hidden layer of logits
#     b6 = variable_on_cpu('b6', [n_hidden_6], tf.random_normal_initializer())
#     h6 = variable_on_cpu('h6', [n_hidden_5, n_hidden_6], tf.random_normal_initializer())
#     layer_6 = tf.add(tf.matmul(layer_5, h6), b6)
# ```
# Applies the weight matrix `h6` and bias `b6` to the output of `layer_5` creating `n_classes` dimensional vectors, the logits.
# 
# The next lines of `BiRNN`
# ```python
#     # Reshape layer_6 from a tensor of shape [n_steps*batch_size, n_hidden_6]
#     # to a tensor of shape [n_steps, batch_size, n_hidden_6]
#     layer_6 = tf.reshape(layer_6, [-1, batch_x_shape[0], n_hidden_6])
# ```
# reshapes `layer_6` to the slightly more useful shape `[n_steps, batch_size, n_hidden_6]`. Note, that this differs from the input in that it is time-major.
# 
# The final line of `BiRNN` returns `layer_6`
# ```python
#     # Return layer_6
#     # Output shape: [n_steps, batch_size, n_hidden_6]
#     return layer_6
# ```

# # Accuracy and Loss

# In accord with [Deep Speech: Scaling up end-to-end speech recognition](http://arxiv.org/abs/1412.5567), the loss function used by our network should be the CTC loss function[[2]](http://www.cs.toronto.edu/~graves/preprint.pdf). Conveniently, this loss function is implemented in TensorFlow. Thus, we can simply make use of this implementation to define our loss.
# 
# To do so we introduce a utility function `calculate_accuracy_and_loss()` that beam search decodes a mini-batch and calculates the loss and accuracy. Next to total and average loss it returns the accuracy, the decoded result and the batch's original Y.

# In[ ]:

def calculate_accuracy_and_loss(batch_set, dropout):
    # Obtain the next batch of data
    batch_x, batch_seq_len, batch_y = batch_set.next_batch()

    # Calculate the logits of the batch using BiRNN
    logits = BiRNN(batch_x, tf.to_int64(batch_seq_len), dropout)
    
    # Compute the CTC loss
    if use_warpctc:
        total_loss = tf.contrib.warpctc.warp_ctc_loss(logits, batch_y, batch_seq_len)
    else:
        total_loss = ctc_ops.ctc_loss(logits, batch_y, batch_seq_len)
    
    # Calculate the average loss across the batch
    avg_loss = tf.reduce_mean(total_loss)
    
    # Beam search decode the batch
    decoded, _ = ctc_ops.ctc_beam_search_decoder(logits, batch_seq_len, merge_repeated=False)
    
    # Compute the edit (Levenshtein) distance 
    distance = tf.edit_distance(tf.cast(decoded[0], tf.int32), batch_y)
    
    # Compute the accuracy 
    accuracy = tf.reduce_mean(distance)

    # Return results to the caller
    return total_loss, avg_loss, distance, accuracy, decoded, batch_y


# The first lines of `calculate_accuracy_and_loss()`
# ```python
# def calculate_accuracy_and_loss(batch_set, dropout):
#     # Obtain the next batch of data
#     batch_x, batch_seq_len, batch_y = batch_set.next_batch()
# ```
# simply obtain the next mini-batch of data.
# 
# The next line
# ```python
#     # Calculate the logits from the BiRNN
#     logits = BiRNN(batch_x, batch_seq_len, dropout)
# ```
# calls `BiRNN()` with a batch of data and does inference on the batch.
# 
# The next few lines
# ```python
#     # Compute the CTC loss
#     total_loss = ctc_ops.ctc_loss(logits, batch_y, batch_seq_len)
#     
#     # Calculate the average loss across the batch
#     avg_loss = tf.reduce_mean(total_loss)
# ```
# calculate the average loss using tensor flow's `ctc_loss` operator. 
# 
# The next lines first beam decode the batch and then compute the accuracy on base of the Levenshtein distance between the decoded batch and the batch's original Y.
# ```python
#     # Beam search decode the batch
#     decoded, _ = ctc_ops.ctc_beam_search_decoder(logits, batch_seq_len, merge_repeated=False)
#     
#     # Compute the edit (Levenshtein) distance 
#     distance = tf.edit_distance(tf.cast(decoded[0], tf.int32), batch_y)
#     
#     # Compute the accuracy 
#     accuracy = tf.reduce_mean(distance)
# ```
# 
# Finally, the calculated total and average losses, the Levenshtein distance and the recognition accuracy are returned, alongside the decoded batch and the original batch_y (which contains the verified transcriptions).
# ```python
#     # Return results to the caller
#     return total_loss, avg_loss, distance, accuracy, decoded, batch_y
# ```

# # Parallel Optimization

# Next we will implement optimization of the DeepSpeech model across GPU's on a single host. This parallel optimization can take on various forms. For example one can use asynchronous updates of the model, synchronous updates of the model, or some combination of the two.

# ## Asynchronous Parallel Optimization

# In asynchronous parallel optimization, for example, one places the model initially in CPU memory. Then each of the $G$ GPU's obtains a mini-batch of data along with the current model parameters. Using this mini-batch each GPU then computes the gradients for all model parameters and sends these gradients back to the CPU when the GPU is done with its mini-batch. The CPU then asynchronously updates the model parameters whenever it recieves a set of gradients from a GPU.

# Asynchronous parallel optimization has several advantages and several disadvantages. One large advantage is throughput. No GPU will every be waiting idle. When a GPU is done processing a mini-batch, it can immediately obtain the next mini-batch to process. It never has to wait on other GPU's to finish their mini-batch. However, this means that the model updates will also be asynchronous which can have problems.
# 
# For example, one may have model parameters $W$ on the CPU and send mini-batch $n$ to GPU 1 and send mini-batch $n+1$ to GPU 2. As processing is asynchronous, GPU 2 may finish before GPU 1 and thus update the CPU's model parameters $W$ with its gradients $\Delta W_{n+1}(W)$, where the subscript $n+1$ identifies the mini-batch and the argument $W$ the location at which the gradient was evaluated. This results in the new model parameters
# 
# $$W + \Delta W_{n+1}(W).$$
# 
# Next GPU 1 could finish with its mini-batch and update the parameters to
# 
# $$W + \Delta W_{n+1}(W) + \Delta W_{n}(W).$$
# 
# The problem with this is that $\Delta W_{n}(W)$ is evaluated at $W$ and not $W + \Delta W_{n+1}(W)$. Hence, the direction of the gradient $\Delta W_{n}(W)$ is slightly incorrect as it is evaluated at the wrong location. This can be counteracted through synchronous updates of model, but this is also problematic.

# ## Synchronous Optimization

# Synchronous optimization solves the problem we saw above. In synchronous optimization, one places the model initially in CPU memory. Then one of the $G$ GPU's is given a mini-batch of data along with the current model parameters. Using the mini-batch the GPU computes the gradients for all model parameters and sends the gradients back to the CPU. The CPU then updates the model parameters and starts the process of sending out the next mini-batch.
# 
# As on can readily see, synchronous optimization does not have the problem we found in the last section, that of incorrect gradients. However, synchronous optimization can only make use of a single GPU at a time. So, when we have a multi-GPU setup, $G > 1$, all but one of the GPU's will remain idle, which is unacceptable. However, there is a third alternative which is combines the advantages of asynchronous and synchronous optimization.

# ## Hybrid Parallel Optimization

# Hybrid parallel optimization combines most of the benifits of asynchronous and synchronous optimization. It allows for multiple GPU's to be used, but does not suffer from the incorrect gradient problem exhibited by asynchronous optimization.
# 
# In hybrid parallel optimization one places the model initially in CPU memory. Then, as in asynchronous optimization, each of the $G$ GPU'S obtains a mini-batch of data along with the current model parameters. Using the mini-batch each of the GPU's then computes the gradients for all model parameters and sends these gradients back to the CPU. Now, in contrast to asynchronous optimization, the CPU waits until each GPU is finished with its mini-batch then takes the mean of all the gradients from the $G$ GPU's and updates the model with this mean gradient.

# <img src="images/Parallelism.png" alt="LSTM" width="600">

# Hybrid parallel optimization has several advantages and few disadvantages. As in asynchronous parallel optimization, hybrid parallel optimization allows for one to use multiple GPU's in parallel. Furthermore, unlike asynchronous parallel optimization, the incorrect gradient problem is not present here. In fact, hybrid parallel optimization performs as if one is working with a single mini-batch which is $G$ times the size of a mini-batch handled by a single GPU. Hoewever, hybrid parallel optimization is not perfect. If one GPU is slower than all the others in completing its mini-batch, all other GPU's will have to sit idle until this straggler finishes with its mini-batch. This hurts throughput. But, if all GPU'S are of the same make and model, this problem should be minimized.
# 
# So, relatively speaking, hybrid parallel optimization seems the have more advantages and fewer disadvantages as compared to both asynchronous and synchronous optimization. So, we will, for our work, use this hybrid model.

# ## Adam Optimization

# In constrast to [Deep Speech: Scaling up end-to-end speech recognition](http://arxiv.org/abs/1412.5567), in which  [Nesterov’s Accelerated Gradient Descent](www.cs.toronto.edu/~fritz/absps/momentum.pdf) was used, we will use the Adam method for optimization[[3](http://arxiv.org/abs/1412.6980)], because, generally, it requires less fine-tuning.

# In[ ]:

def create_optimizer():
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                       beta1=beta1,
                                       beta2=beta2,
                                       epsilon=epsilon)
    return optimizer


# ## Towers

# In order to properly make use of multiple GPU's, one must introduce new abstractions, not present when using a single GPU, that facilitate the multi-GPU use case.
# 
# In particular, one must introduce a means to isolate the inference and gradient calculations on the various GPU's. The abstraction we intoduce for this purpose is called a 'tower'. A tower is specified by two properties:
# * **Scope** - A scope, as provided by `tf.name_scope()`, is a means to isolate the operations within a tower. For example, all operations within "tower 0" could have their name prefixed with `tower_0/`.
# *  **Device** - A hardware device, as provided by `tf.device()`, on which all operations within the tower execute. For example, all operations of "tower 0" could execute on the first GPU `tf.device('/gpu:0')`.

# As we are introducing one tower for each GPU, first we must determine how many GPU's are available

# In[ ]:

# Get a list of the available gpu's ['/gpu:0', '/gpu:1'...]
available_devices = get_available_gpus()

# If there are no GPU's use the CPU
if 0 == len(available_devices):
    available_devices = ['/cpu:0']


# With this preliminary step out of the way, we can for each GPU introduce a tower for which's batch we calculate 
#  
#  * the CTC decodings ```decoded```,
#  * the (total) loss against the outcome (Y) ```total_loss```, 
#  * the loss averaged over the whole batch ```avg_loss```,
#  * the optimization gradient (computed based on the averaged loss),
#  * the Levenshtein distances between the decodings and their transcriptions ```distance```,
#  * the accuracy of the outcome averaged over the whole batch ```accuracy``` 
#  
# and retain the original ```labels``` (Y).
#  
# ```decoded```, ```labels```, the optimization gradient, ```distance```, ```accuracy```, ```total_loss``` and ```avg_loss``` are collected into the corresponding arrays ```tower_decodings, tower_labels, tower_gradients, tower_distances, tower_accuracies, tower_total_losses, tower_avg_losses``` (dimension 0 being the tower).
# 
# Finally this new method `get_tower_results()` will return those tower arrays.
# In case of ```tower_accuracies``` and ```tower_avg_losses```, it will return the averaged values instead of the arrays.

# In[ ]:

def get_tower_results(batch_set, optimizer=None):  
    # Tower labels to return
    tower_labels = []
    # Tower decodings to return
    tower_decodings = []
    # Tower distances to return
    tower_distances = []
    # Tower total batch losses to return
    tower_total_losses = []
    
    # Tower gradients to return
    tower_gradients = []
    
    # To calculate the mean of the accuracies
    tower_accuracies = []
    
    # To calculate the mean of the losses
    tower_avg_losses = []
    
    # Loop over available_devices
    for i in xrange(len(available_devices)):
        # Execute operations of tower i on device i
        with tf.device(available_devices[i]):
            # Create a scope for all operations of tower i
            with tf.name_scope('tower_%d' % i) as scope:
                # Calculate the avg_loss and accuracy and retrieve the decoded 
                # batch along with the original batch's labels (Y) of this tower
                total_loss, avg_loss, distance, accuracy, decoded, labels =                     calculate_accuracy_and_loss(batch_set, 0.0 if optimizer is None else dropout_rate)
                                
                # Allow for variables to be re-used by the next tower
                tf.get_variable_scope().reuse_variables()
                
                # Retain tower's labels (Y)
                tower_labels.append(labels)
                
                # Retain tower's decoded batch
                tower_decodings.append(decoded)
                
                # Retain tower's distances
                tower_distances.append(distance)
                
                # Retain tower's total losses
                tower_total_losses.append(total_loss)

                if optimizer is not None:
                    # Compute gradients for model parameters using tower's mini-batch
                    gradients = optimizer.compute_gradients(avg_loss)

                    # Retain tower's gradients
                    tower_gradients.append(gradients)
                    
                # Retain tower's accuracy
                tower_accuracies.append(accuracy)
                
                # Retain tower's avg losses
                tower_avg_losses.append(avg_loss)

    # Return the results tuple, the gradients, and the means of accuracies and losses
    return         (tower_labels, tower_decodings, tower_distances, tower_total_losses),         tower_gradients,         tf.reduce_mean(tower_accuracies, 0),         tf.reduce_mean(tower_avg_losses, 0)


# Next we want to average the gradients obtained from the GPU's.

# We compute the average the gradients obtained from the GPU's for each variable in the function `average_gradients()`

# In[ ]:

def average_gradients(tower_gradients):
    # List of average gradients to return to the caller
    average_grads = []
    
    # Loop over gradient/variable pairs from all towers
    for grad_and_vars in zip(*tower_gradients):
        # Introduce grads to store the gradients for the current variable
        grads = []
        
        # Loop over the gradients for the current variable
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)
            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)
            
        # Average over the 'tower' dimension
        grad = tf.concat(0, grads)
        grad = tf.reduce_mean(grad, 0)
        
        # Create a gradient/variable tuple for the current variable with its average gradient
        grad_and_var = (grad, grad_and_vars[0][1])
        
        # Add the current tuple to average_grads
        average_grads.append(grad_and_var)
    
    #Return result to caller
    return average_grads


# Note also that this code acts as a syncronization point as it requires all GPU's to be finished with their mini-batch before it can run to completion.
# 
# Now next we introduce a function to apply the averaged gradients to update the model's paramaters on the CPU

# In[ ]:

def apply_gradients(optimizer, average_grads):
    apply_gradient_op = optimizer.apply_gradients(average_grads)
    return apply_gradient_op


# # Logging

# We introduce a function for logging a tensor variable's current state.
# It logs scalar values for the mean, standard deviation, minimum and maximum.
# Furthermore it logs a histogram of its state and (if given) of an optimization gradient.

# In[ ]:

def log_variable(variable, gradient=None):
    name = variable.name
    mean = tf.reduce_mean(variable)
    tf.scalar_summary(name + '/mean', mean)
    tf.scalar_summary(name + '/sttdev', tf.sqrt(tf.reduce_mean(tf.square(variable - mean))))
    tf.scalar_summary(name + '/max', tf.reduce_max(variable))
    tf.scalar_summary(name + '/min', tf.reduce_min(variable))
    tf.histogram_summary(name, variable)
    if gradient is not None:
        if isinstance(gradient, tf.IndexedSlices):
            grad_values = gradient.values
        else:
            grad_values = gradient
        if grad_values is not None:
            tf.histogram_summary(name + "/gradients", grad_values)


# Let's also introduce a helper function for logging collections of gradient/variable tuples.

# In[ ]:

def log_grads_and_vars(grads_and_vars):
    for gradient, variable in grads_and_vars:
        log_variable(variable, gradient=gradient)


# Finally we define the top directory for all logs and our current log sub-directory of it.
# We also add some log helpers.

# In[ ]:

logs_dir = os.environ.get('ds_logs_dir', 'logs')
log_dir = '%s/%s' % (logs_dir, time.strftime("%Y%m%d-%H%M%S"))

def get_git_revision_hash():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip()

def get_git_branch():
    return subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).strip()


# # Helpers

# Let's introduce a routine that prints a WER report under a given caption. It'll print the given ```mean``` WER plus summaries of the top ten lowest loss items of the given array of WER result tuples (only items with WER!=0 and ordered by their WER).

# In[ ]:

def calculate_and_print_wer_report(caption, results_tuple):
    
    items = zip(*results_tuple)

    count = len(items)
    mean_wer = 0.0
    for i in xrange(count):
        item = items[i]
        # If distance > 0 we know that there is a WER > 0 and have to calculate it
        if item[2] > 0:
            # Replacing accuracy tuple entry by the WER
            item = items[i] = (item[0], item[1], wer(item[0], item[1]), item[3])
            mean_wer = mean_wer + item[2]
    
    # Getting the mean WER from the accumulated one
    mean_wer = mean_wer / float(count)
    
    # Filter out all items with WER=0
    items = [a for a in items if a[2] > 0]
    
    # Order the remaining items by their loss (lowest loss on top)
    items.sort(key=lambda a: a[3])

    # Take only the first report_count items
    items = items[:report_count]

    # Order this top ten items by their WER (lowest WER on top)
    items.sort(key=lambda a: a[2])

    print "%s WER: %f" % (caption, mean_wer)
    for a in items:
        print "-" * 80
        print "    WER:    %f" % a[2]
        print "    loss:   %f" % a[3]
        print "    source: \"%s\"" % a[0]
        print "    result: \"%s\"" % a[1] 

    return mean_wer


# Another routine will help collecting partial results for the WER reports. The ```results_tuple``` is composed of an array of the original labels, an array of the corresponding decodings, an array of the corrsponding distances and an array of the corresponding losses. ```returns``` is built up in a similar way, containing just the unprocessed results of one ```session.run``` call (effectively of one batch). Labels and decodings are converted to text before splicing them into their corresponding results_tuple lists. In the case of decodings, for now we just pick the first available path.

# In[ ]:

def collect_results(results_tuple, returns):
    # Each of the arrays within results_tuple will get extended by a batch of each available device
    for i in xrange(len(available_devices)):
        # Collect the labels
        results_tuple[0].extend(sparse_tensor_value_to_texts(returns[0][i]))
        
        # Collect the decodings - at the moment we default to the first one 
        results_tuple[1].extend(sparse_tensor_value_to_texts(returns[1][i][0]))
        
        # Collect the distances
        results_tuple[2].extend(returns[2][i])
        
        # Collect the losses
        results_tuple[3].extend(returns[3][i])


# For reporting we also need a standard way to do time measurements.

# In[ ]:

def stopwatch(start_duration=0):
    """
    This function will toggle a stopwatch. 
    The first call starts it, second call stops it, third call continues it etc.
    So if you want to measure the accumulated time spent in a certain area of the code,
    you can surround that code by stopwatch-calls like this:
    fun_time = 0 # initializes a stopwatch
    [...]
    for i in xrange(10):
      [...]
      fun_time = stopwatch(fun_time) # starts/continues the stopwatch - fun_time is now a point in time (again)
      fun()
      fun_time = stopwatch(fun_time) # pauses the stopwatch - fun_time is now a duration
    [...]
    # the following line only makes sense after an even call of "fun_time = stopwatch(fun_time)"
    print "Time spent in fun():", format_duration(fun_time) 
    """
    if start_duration == 0:
        return datetime.datetime.utcnow()
    else:
        return datetime.datetime.utcnow() - start_duration

def format_duration(duration):
    """Formats the result of an even stopwatch call as hours:minutes:seconds"""
    m, s = divmod(duration.seconds, 60)
    h, m = divmod(m, 60)
    return "%d:%02d:%02d" % (h, m, s)


# # Execution
# 
# To run our different graphs in separate sessions, we first need to create some common infrastructure.
# 
# At first we introduce the functions `read_data_sets` and `read_data_set` to read in data sets.
# The first returns a `DataSets` object of the selected importer, containing all available sets.
# The latter takes the name of the required data set (`'train'`, `'dev'` or `'test'`) as string and returns the respective set.

# In[ ]:

def read_data_sets():
    # Obtain all the data sets
    return ds_importer_module.read_data_sets(ds_dataset_path,                                              train_batch_size,                                              dev_batch_size,                                              test_batch_size,                                              n_input,                                              n_context,                                              limit_dev=limit_dev,                                              limit_test=limit_test,                                              limit_train=limit_train)

def read_data_set(set_name):
    # Obtain all the data sets
    data_sets = read_data_sets()
    # Pick the train, dev, or test data set from it
    return getattr(data_sets, set_name)


# The most important data structure that will be shared among the following routines is a so called `execution context`. It's a tuple with four elements: The graph, the data set (one of train/dev/test), the top level graph entry point tuple from `get_tower_results()` and a saver object for persistence.
# 
# Let's at first introduce the construction routine for an execution context. It takes the data set's name as string ("train", "dev" or "test") and returns the execution context tuple.
# 
# An execution context tuple is of the form `(graph, data_set, tower_results, saver)` when not training. `graph` is the `tf.Graph` in which the operators reside. `data_set` is the selected data set (train, dev, or test). `tower_results` is the result of a call to `get_tower_results()`. `saver` is a `tf.train.Saver` which can be used to save the model.
# 
# When training an execution context is of the form `(graph, data_set, tower_results, saver, apply_gradient_op, merged, writer)`. The first four items are the same as in the above case. `apply_gradient_op` is an operator that applies the gradents to the learned parameters. `merged` contains all summaries for tensorboard. Finally, `writer` is the `tf.train.SummaryWriter` used to write summaries for tensorboard.

# In[ ]:

def create_execution_context(set_name):
    graph = tf.Graph()
    with graph.as_default():
        # Set the global random seed for determinism
        tf.set_random_seed(random_seed)
        
        # Get the required data set
        data_set = read_data_set(set_name)
        
        # Define bool to indicate if data_set is the training set
        is_train = set_name == 'train'
        
        # If training create optimizer
        optimizer = create_optimizer() if is_train else None
        
        # Get the data_set specific graph end-points
        tower_results = get_tower_results(data_set, optimizer=optimizer)
        
        if is_train:
            # Average tower gradients
            avg_tower_gradients = average_gradients(tower_results[1])

            # Add logging of averaged gradients
            if log_variables:
                log_grads_and_vars(avg_tower_gradients)

            # Apply gradients to modify the model
            apply_gradient_op = apply_gradients(optimizer, avg_tower_gradients)

        # Create a saver to checkpoint the model
        saver = tf.train.Saver(tf.all_variables())
        
        if is_train:
            # Prepare tensor board logging
            merged = tf.merge_all_summaries()
            writer = tf.train.SummaryWriter(log_dir, graph)
            return (graph, data_set, tower_results, saver, apply_gradient_op, merged, writer)
        else:
            return (graph, data_set, tower_results, saver)


# Now let's introduce a routine for starting an execution context. By passing in the execution context and the file path of the model, it will
#  - create a new session using the execution model's graph,
#  - load (restore) the model from the file path into it,
#  - start the associated queue and runner threads.
#  
# Finally it will return the new session.

# In[ ]:

def start_execution_context(execution_context, model_path=None):
    # Obtain the Graph in which to execute
    graph = execution_context[0]
    
    # Create a new session and load the execution context's graph into it
    session = tf.Session(config=session_config, graph=graph)
    
    # Set graph as the default Graph
    with graph.as_default():
        if model_path is None:
            # Init all variables for first use
            session.run(tf.initialize_all_variables())
        else:
            # Loading the model into the session
            execution_context[3].restore(session, model_path)

        # Create Coordinator to manage threads
        coord = tf.train.Coordinator()

        # Start queue runner threads
        managed_threads = tf.train.start_queue_runners(sess=session, coord=coord)

        # Start importer's queue threads
        managed_threads = managed_threads + execution_context[1].start_queue_threads(session, coord)

    return session, coord, managed_threads


# The following helper method persists the contained model to disk and returns the model's filename, constructed from `checkpoint_path` and `global_step`

# In[ ]:

def persist_model(execution_context, session, checkpoint_path, global_step):
    # Saving session's model into checkpoint dir
    return execution_context[3].save(session, checkpoint_path, global_step=global_step)


# The following helper method stops an execution context. 
# Before closing the provided `session`, it will persist the contained model to disk. 
# The model's filename will be returned.

# In[ ]:

def stop_execution_context(execution_context, session, coord, managed_threads, checkpoint_path=None, global_step=None):
    
    # If the model is not persisted, we'll return 'None'
    hibernation_path = None
    
    if checkpoint_path and global_step:
        # Saving session's model into checkpoint dir
        hibernation_path = persist_model(execution_context, session, checkpoint_path, global_step)
        
    # Close importer's queue
    execution_context[1].close_queue(session)
    
    # Request managed threads stop
    coord.request_stop()
    
    # Wait until managed threads stop
    coord.join(managed_threads)

    # Free all allocated resources of the session
    session.close()
    
    return hibernation_path


# Now let's introduce the main routine for training and inference.
# 
# It takes a started execution context (given by `execution_context`) a `Session` (`session`), an optional epoch index (`epoch`) and a flag (`query_report`) which indicates whether to calculate the WER report data or not.
# 
# Its main duty is to iterate over all batches and calculate the mean loss. If a non-negative epoch is provided, it will also optimize the parameters. If `query_report` is `False`, the default, it will return a tuple which contains the mean loss. If `query_report` is `True`, the mean accuracy and individual results are also included in the returned tuple.

# In[ ]:

def calculate_loss_and_report(execution_context, session, epoch=-1, query_report=False):
    
    # An epoch of -1 means no train run
    do_training = epoch >= 0
    
    # Unpack variables
    if do_training:
        graph, data_set, tower_results, saver, apply_gradient_op, merged, writer = execution_context
    else:
        graph, data_set, tower_results, saver = execution_context
    results_params, _, avg_accuracy, avg_loss = tower_results
    
    batches_per_device = ceil(float(data_set.total_batches) / len(available_devices))

    total_loss = 0.0
    params = OrderedDict()
    params['avg_loss'] = avg_loss
    
    # Facilitate a train run
    if do_training:
        params['apply_gradient_op'] = apply_gradient_op
        if log_variables:
            params['merged'] = merged
    
    # Requirements to display a WER report
    if query_report:
        # Reset accuracy
        total_accuracy = 0.0
        # Create report results tuple
        report_results = ([],[],[],[])
        # Extend the session.run parameters
        params['sample_results'] = [results_params, avg_accuracy]
        
    # Get the index of each of the session fetches so we can recover the results more easily
    param_idx = dict(zip(params.keys(), range(len(params))))
    params = params.values()

    # Loop over the batches
    for batch in range(int(batches_per_device)):
        extra_params = { }
        if do_training and do_fulltrace:
            loss_run_metadata            = tf.RunMetadata()
            extra_params['options']      = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            extra_params['run_metadata'] = loss_run_metadata

        # Compute the batch
        result = session.run(params, **extra_params)
        
        # Add batch to loss
        total_loss += result[param_idx['avg_loss']]
        
        if do_training:
            # Log all variable states in current step
            step = epoch * data_set.total_batches + batch * len(available_devices)
            if log_variables:
                writer.add_summary(result[param_idx['merged']], step)
            if do_fulltrace:
                writer.add_run_metadata(loss_run_metadata, 'loss_epoch%d_batch%d'   % (epoch, batch))
            writer.flush()

        if query_report:
            sample_results = result[param_idx['sample_results']]
            # Collect individual sample results
            collect_results(report_results, sample_results[0])
            # Add batch to total_accuracy
            total_accuracy += sample_results[1]
            
    # Returning the batch set result tuple
    loss = total_loss / batches_per_device
    if query_report:
        return (loss, total_accuracy / batches_per_device, report_results)
    else:
        return (loss, None, None)


# The following routine will print a report from a provided batch set result tuple. It takes a caption for titling the output plus the batch set result tuple. If the batch set result tuple contains accuracy and a report results tuple, a complete WER report will be calculated, printed and its mean WER returned. Otherwise it will just print the loss and return `None`.

# In[ ]:

def print_report(caption, batch_set_result):
    
    # Unpacking batch set result tuple
    loss, accuracy, results_tuple = batch_set_result
    
    # We always have a loss value
    title = caption + " loss=" + "{:.9f}".format(loss)
    
    mean_wer = None
    if accuracy is not None and results_tuple is not None:
        title += " avg_cer=" + "{:.9f}".format(accuracy)
        mean_wer = calculate_and_print_wer_report(title, results_tuple)
    else:
        print title

    return mean_wer


# Let's also introduce a routine that facilitates obtaining results from a data set (given by its name in `set_name`) - from execution context creation to closing the session. 
# If a model's filename is provided by `model_path`, it will initialize the session by loading the given model into it. 
# It will return the loss and - if `query_report=True` - also the accuracy and the report results tuple.

# In[ ]:

def run_set(set_name, model_path=None, query_report=False):
    
    # Creating the execution context
    execution_context = create_execution_context(set_name)
    
    # Starting the execution context 
    session, coord, managed_threads = start_execution_context(execution_context, model_path=model_path)
    
    # Applying the batches
    batch_set_result = calculate_loss_and_report(execution_context, session, query_report=query_report)
    
    # Cleaning up
    stop_execution_context(execution_context, session, coord, managed_threads)

    # Returning batch_set_result from calculate_loss_and_report
    return batch_set_result


# # Training

# Now, as we have prepared all the apropos operators and methods, we can create the method which trains the network.

# In[ ]:

def train():
    print "STARTING Optimization\n"
    global_time = stopwatch()
    global_train_time = 0
    
    # Creating the training execution context
    train_context = create_execution_context('train')

    # Init recent word error rate levels
    train_wer = 0.0
    dev_wer = 0.0

    hibernation_path = None

    # Possibly restore checkpoint
    start_epoch = 0
    if restore_checkpoint:
        checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
        if checkpoint and checkpoint.model_checkpoint_path:
            hibernation_path = checkpoint.model_checkpoint_path
            start_epoch = int(checkpoint.model_checkpoint_path.split('-')[-1])
            print 'Resuming training from epoch %d' % (start_epoch + 1)
    
    # Loop over the data set for training_epochs epochs
    for epoch in range(start_epoch, epochs):

        print "STARTING Epoch", '%04d' % (epoch)
        
        if epoch == 0 or hibernation_path is not None:
            if hibernation_path is not None:
                print "Resuming training session from", "%s" % hibernation_path, "..."
            session, coord, managed_threads = start_execution_context(train_context, hibernation_path)
        # The next loop should not load the model, unless it got set again in the meantime (by validation)
        hibernation_path = None
        
        overall_time = stopwatch()
        train_time = 0

        # Determine if we want to display, validate, checkpoint on this iteration
        is_display_step = display_step > 0 and ((epoch + 1) % display_step == 0 or epoch == epochs - 1)
        is_validation_step = validation_step > 0 and (epoch > 0 and (epoch + 1) % validation_step == 0)
        is_checkpoint_step = (checkpoint_step > 0 and epoch > 0 and (epoch + 1) % checkpoint_step == 0) or                              epoch == epochs - 1

        print "Training model..."
        global_train_time = stopwatch(global_train_time)
        train_time = stopwatch(train_time)
        result = calculate_loss_and_report(train_context, session, epoch=epoch, query_report=is_display_step)
        global_train_time = stopwatch(global_train_time)
        train_time = stopwatch(train_time)

        result = print_report("Training", result)
        # If there was a WER calculated, we keep it
        if result is not None:
            train_wer = result
        
        # Checkpoint step (Validation also checkpoints)
        if is_checkpoint_step and not is_validation_step:
            print "Hibernating training session into directory", "%s" % checkpoint_dir
            checkpoint_path = os.path.join(checkpoint_dir, 'model.ckpt')
            # Saving session's model into checkpoint path
            persist_model(train_context, session, checkpoint_path, epoch)
        # Validation step
        if is_validation_step:
            print "Hibernating training session into directory", "%s" % checkpoint_dir
            checkpoint_path = os.path.join(checkpoint_dir, 'model.ckpt')
            # If the hibernation_path is set, the next epoch loop will load the model
            hibernation_path = stop_execution_context(                 train_context, session, coord, managed_threads, checkpoint_path=checkpoint_path, global_step=epoch)
            
            # Validating the model in a fresh session
            print "Validating model..."
            result = run_set('dev', model_path=hibernation_path, query_report=True)
            result = print_report("Validation", result)
            # If there was a WER calculated, we keep it
            if result is not None:
                dev_wer = result

        overall_time = stopwatch(overall_time)
            
        print "FINISHED Epoch", '%04d' % (epoch),             "  Overall epoch time:", format_duration(overall_time),             "  Training time:", format_duration(train_time)
        print

    # If the last iteration step was no validation, we still have to save the model
    if hibernation_path is None:
        hibernation_path = stop_execution_context(             train_context, session, coord, managed_threads, checkpoint_path=checkpoint_path, global_step=epoch)
    
    # Indicate optimization has concluded
    print "FINISHED Optimization",         "  Overall time:", format_duration(stopwatch(global_time)),         "  Training time:", format_duration(global_train_time)
    print
    
    return train_wer, dev_wer, hibernation_path


# As everything is prepared, we are now able to do the training.

# In[ ]:

# Define CPU as device on which the muti-gpu training is orchestrated
with tf.device('/cpu:0'):
    # Take start time for time measurement
    time_started = datetime.datetime.utcnow()
    
    # Train the network
    last_train_wer, last_dev_wer, hibernation_path = train()
    
    # Take final time for time measurement
    time_finished = datetime.datetime.utcnow()
    
    # Calculate duration in seconds
    duration = time_finished - time_started
    duration = duration.days * 86400 + duration.seconds
    
    # Finally the model is tested against some unbiased data-set
    print "Testing model"
    result = run_set('test', model_path=hibernation_path, query_report=True)
    test_wer = print_report("Test", result)


# Finally, we restore the trained variables into a simpler graph that we can export for serving.

# In[ ]:

# Don't export a model if no export directory has been set
if export_dir:
    with tf.device('/cpu:0'):
        tf.reset_default_graph()
        session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))

        # Run inference
        
        # Input tensor will be of shape [batch_size, n_steps, n_input + 2*n_input*n_context]
        input_tensor = tf.placeholder(tf.float32, [None, None, n_input + 2*n_input*n_context])

        # Calculate input sequence length. This is done by tiling n_steps, batch_size times.
        # If there are multiple sequences, it is assumed they are padded with zeros to be of
        # the same length.
        n_items  = tf.slice(tf.shape(input_tensor), [0], [1])
        n_steps = tf.slice(tf.shape(input_tensor), [1], [1])
        seq_length = tf.tile(n_steps, n_items)

        # Calculate the logits of the batch using BiRNN
        logits = BiRNN(input_tensor, tf.to_int64(seq_length), 0)

        # Beam search decode the batch
        decoded, _ = ctc_ops.ctc_beam_search_decoder(logits, seq_length, merge_repeated=False)
        decoded = tf.convert_to_tensor(
            [tf.sparse_tensor_to_dense(sparse_tensor) for sparse_tensor in decoded])

        # TODO: Transform the decoded output to a string

        # Create a saver and exporter using variables from the above newly created graph
        saver = tf.train.Saver(tf.all_variables())
        model_exporter = exporter.Exporter(saver)
        
        # Restore variables from training checkpoint
        # TODO: This restores the most recent checkpoint, but if we use validation to counterract
        #       over-fitting, we may want to restore an earlier checkpoint.
        checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
        saver.restore(session, checkpoint.model_checkpoint_path)
        print 'Restored checkpoint at training epoch %d' % (int(checkpoint.model_checkpoint_path.split('-')[-1]) + 1)

        # Initialise the model exporter and export the model
        model_exporter.init(session.graph.as_graph_def(),
                            named_graph_signatures = {
                                'inputs': exporter.generic_signature(
                                    { 'input': input_tensor }),
                                'outputs': exporter.generic_signature(
                                    { 'outputs': decoded})})
        if remove_export:
            actual_export_dir = os.path.join(export_dir, '%08d' % export_version)
            if os.path.isdir(actual_export_dir):
                print 'Removing old export'
                shutil.rmtree(actual_export_dir)
        try:
            model_exporter.export(export_dir, tf.constant(export_version), session)
            print 'Model exported at %s' % (export_dir)
        except RuntimeError:
            print  sys.exc_info()[1]


# # Logging Hyper Parameters and Results

# Now, as training and test are done, we persist the results alongside with the involved hyper parameters for further reporting.

# In[ ]:

data_sets = read_data_sets()

with open('%s/%s' % (log_dir, 'hyper.json'), 'w') as dump_file:
    json.dump({         'context': {             'time_started': time_started.isoformat(),             'time_finished': time_finished.isoformat(),             'git_hash': get_git_revision_hash(),             'git_branch': get_git_branch()         },         'parameters': {             'learning_rate': learning_rate,             'beta1': beta1,             'beta2': beta2,             'epsilon': epsilon,             'epochs': epochs,             'train_batch_size': train_batch_size,             'dev_batch_size': dev_batch_size,             'test_batch_size': test_batch_size,             'validation_step': validation_step,             'dropout_rate': dropout_rate,             'relu_clip': relu_clip,             'n_input': n_input,             'n_context': n_context,             'n_hidden_1': n_hidden_1,             'n_hidden_2': n_hidden_2,             'n_hidden_3': n_hidden_3,             'n_hidden_5': n_hidden_5,             'n_hidden_6': n_hidden_6,             'n_cell_dim': n_cell_dim,             'n_character': n_character,             'total_batches_train': data_sets.train.total_batches,             'total_batches_validation': data_sets.dev.total_batches,             'total_batches_test': data_sets.test.total_batches,             'data_set': {                 'name': ds_importer             }         },         'results': {             'duration': duration,             'last_train_wer': last_train_wer,             'last_validation_wer': last_dev_wer,             'test_wer': test_wer         }     }, dump_file, sort_keys=True, indent = 4)


# Let's also re-populate a central JS file, that contains all the dumps at once.

# In[ ]:

merge_logs(logs_dir)
maybe_publish()

