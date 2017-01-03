# Geometric Constants

This is about several constants related to the geometry of the network.

## n_steps
The network views each speech sample as a sequence of time-slices <img src="https://rawgit.com/mozilla/DeepSpeech/issue313/doc/svgs//537fe6f40a03a282859c43fb7aa2874f.svg?invert_in_darkmode" align=middle width=24.229095pt height=34.27314pt/> of
length <img src="https://rawgit.com/mozilla/DeepSpeech/issue313/doc/svgs//3f623f5f39889e228ea730e85d8200bd.svg?invert_in_darkmode" align=middle width=26.722575pt height=29.12679pt/>. As the speech samples vary in length, we know that <img src="https://rawgit.com/mozilla/DeepSpeech/issue313/doc/svgs//3f623f5f39889e228ea730e85d8200bd.svg?invert_in_darkmode" align=middle width=26.722575pt height=29.12679pt/>
need not equal <img src="https://rawgit.com/mozilla/DeepSpeech/issue313/doc/svgs//1741c589c4fdb782aa008664d3cf9bf7.svg?invert_in_darkmode" align=middle width=28.173255pt height=29.12679pt/> for <img src="https://rawgit.com/mozilla/DeepSpeech/issue313/doc/svgs//4d56de33ada8539cfdcff54baea8310e.svg?invert_in_darkmode" align=middle width=35.19351pt height=22.74591pt/>. For each batch, BRNN in TensorFlow needs
to know `n_steps` which is the maximum <img src="https://rawgit.com/mozilla/DeepSpeech/issue313/doc/svgs//3f623f5f39889e228ea730e85d8200bd.svg?invert_in_darkmode" align=middle width=26.722575pt height=29.12679pt/> for the batch.

## n_input
Each of the at maximum `n_steps` vectors is a vector of MFCC features of a
time-slice of the speech sample. We will make the number of MFCC features
dependent upon the sample rate of the data set. Generically, if the sample rate
is 8kHz we use 13 features. If the sample rate is 16kHz we use 26 features...
We capture the dimension of these vectors, equivalently the number of MFCC
features, in the variable `n_input`

## n_context
As previously mentioned, the BRNN is not simply fed the MFCC features of a given
time-slice. It is fed, in addition, a context of <img src="https://rawgit.com/mozilla/DeepSpeech/issue313/doc/svgs//3b3d517eaf7a36186d0f2c9bf921e14f.svg?invert_in_darkmode" align=middle width=88.45122pt height=24.56553pt/> frames on
either side of the frame in question. The number of frames in this context is
captured in the variable `n_context`

Next we will introduce constants that specify the geometry of some of the
non-recurrent layers of the network. We do this by simply specifying the number
of units in each of the layers

## n_hidden_1, n_hidden_2, n_hidden_5
`n_hidden_1` is the number of units in the first layer, `n_hidden_2` the number
of units in the second, and  `n_hidden_5` the number in the fifth. We haven't
forgotten about the third or sixth layer. We will define their unit count below.

A LSTM BRNN consists of a pair of LSTM RNN's.
One LSTM RNN that works "forward in time"

<img src="../images/LSTM3-chain.png" alt="LSTM" width="800">

and a second LSTM RNN that works "backwards in time"

<img src="../images/LSTM3-chain.png" alt="LSTM" width="800">

The dimension of the cell state, the upper line connecting subsequent LSTM units,
is independent of the input dimension and the same for both the forward and
backward LSTM RNN.

## n_cell_dim
Hence, we are free to choose the dimension of this cell state independent of the
input dimension. We capture the cell state dimension in the variable `n_cell_dim`.

## n_hidden_3
The number of units in the third layer, which feeds in to the LSTM, is
determined by `n_cell_dim` as follows
```python
n_hidden_3 = 2 * n_cell_dim
```

## n_character
The variable `n_character` will hold the number of characters in the target
language plus one, for the <img src="https://rawgit.com/mozilla/DeepSpeech/issue313/doc/svgs//2566b4f99fd69f63a00f0abbc8415767.svg?invert_in_darkmode" align=middle width=44.315865pt height=22.74591pt/>.
For English it is the cardinality of the set
<img src="https://rawgit.com/mozilla/DeepSpeech/issue313/doc/svgs//620afbde0635e244e162a8b0a2bcb9cb.svg?invert_in_darkmode" align=middle width=270.722595pt height=24.56553pt/>
we referred to earlier.

## n_hidden_6
The number of units in the sixth layer is determined by `n_character` as follows
```python
n_hidden_6 = n_character
```
