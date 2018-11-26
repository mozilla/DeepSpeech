Geometric Constants
===================

This is about several constants related to the geometry of the network.

n_steps
-------
The network views each speech sample as a sequence of time-slices :math:`x^{(i)}_t` of
length :math:`T^{(i)}`. As the speech samples vary in length, we know that :math:`T^{(i)}`
need not equal :math:`T^{(j)}` for :math:`i \ne j`. For each batch, BRNN in TensorFlow needs
to know ``n_steps`` which is the maximum :math:`T^{(i)}` for the batch.

n_input
-------
Each of the at maximum ``n_steps`` vectors is a vector of MFCC features of a
time-slice of the speech sample. We will make the number of MFCC features
dependent upon the sample rate of the data set. Generically, if the sample rate
is 8kHz we use 13 features. If the sample rate is 16kHz we use 26 features...
We capture the dimension of these vectors, equivalently the number of MFCC
features, in the variable ``n_input``.

n_context
---------
As previously mentioned, the BRNN is not simply fed the MFCC features of a given
time-slice. It is fed, in addition, a context of :math:`C \in \{5, 7, 9\}` frames on
either side of the frame in question. The number of frames in this context is
captured in the variable ``n_context``.

Next we will introduce constants that specify the geometry of some of the
non-recurrent layers of the network. We do this by simply specifying the number
of units in each of the layers.

n_hidden_1, n_hidden_2, n_hidden_5
----------------------------------
``n_hidden_1`` is the number of units in the first layer, ``n_hidden_2`` the number
of units in the second, and  ``n_hidden_5`` the number in the fifth. We haven't
forgotten about the third or sixth layer. We will define their unit count below.

A LSTM BRNN consists of a pair of LSTM RNN's.
One LSTM RNN that works "forward in time":

.. image:: ../images/LSTM3-chain.png
    :alt: Image shows a diagram of a recurrent neural network with LSTM cells, with arrows depicting the flow of data from earlier time steps to later timesteps within the RNN.

and a second LSTM RNN that works "backwards in time":

.. image:: ../images/LSTM3-chain-backwards.png
    :alt: Image shows a diagram of a recurrent neural network with LSTM cells, this time with data flowing from later time steps to earlier timesteps within the RNN.

The dimension of the cell state, the upper line connecting subsequent LSTM units,
is independent of the input dimension and the same for both the forward and
backward LSTM RNN.

n_cell_dim
----------
Hence, we are free to choose the dimension of this cell state independent of the
input dimension. We capture the cell state dimension in the variable ``n_cell_dim``.

n_hidden_3
----------
The number of units in the third layer, which feeds in to the LSTM, is
determined by ``n_cell_dim`` as follows

.. code:: python

    n_hidden_3 = 2 * n_cell_dim

n_character
-----------
The variable ``n_character`` will hold the number of characters in the target
language plus one, for the :math:`blank`.
For English it is the cardinality of the set

.. math::
    \{a,b,c, . . . , z, space, apostrophe, blank\}

we referred to earlier.

n_hidden_6
----------
The number of units in the sixth layer is determined by ``n_character`` as follows:

.. code:: python

    n_hidden_6 = n_character

