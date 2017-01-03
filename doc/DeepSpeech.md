
# Introduction

In this notebook we will reproduce the results of
[Deep Speech: Scaling up end-to-end speech recognition](http://arxiv.org/abs/1412.5567).
The core of the system is a bidirectional recurrent neural network (BRNN)
trained to ingest speech spectrograms and generate English text transcriptions.

Let a single utterance <img src="https://rawgit.com/mozilla/DeepSpeech/issue313/doc/svgs//332cc365a4987aacce0ead01b8bdcc0b.svg?invert_in_darkmode" align=middle width=9.359955pt height=14.10255pt/> and label <img src="https://rawgit.com/mozilla/DeepSpeech/issue313/doc/svgs//deceeaf6940a8c7a5a02373728002b0f.svg?invert_in_darkmode" align=middle width=8.61696pt height=14.10255pt/> be sampled from a training set
<img src="https://rawgit.com/mozilla/DeepSpeech/issue313/doc/svgs//7b79365abe0c9abfa5d858b170803f9c.svg?invert_in_darkmode" align=middle width=224.067195pt height=29.12679pt/>.
Each utterance, <img src="https://rawgit.com/mozilla/DeepSpeech/issue313/doc/svgs//ad769e751231d17313953f80471b27a4.svg?invert_in_darkmode" align=middle width=24.229095pt height=29.12679pt/> is a time-series of length <img src="https://rawgit.com/mozilla/DeepSpeech/issue313/doc/svgs//3f623f5f39889e228ea730e85d8200bd.svg?invert_in_darkmode" align=middle width=26.722575pt height=29.12679pt/>
where every time-slice is a vector of audio features,
<img src="https://rawgit.com/mozilla/DeepSpeech/issue313/doc/svgs//537fe6f40a03a282859c43fb7aa2874f.svg?invert_in_darkmode" align=middle width=24.229095pt height=34.27314pt/> where <img src="https://rawgit.com/mozilla/DeepSpeech/issue313/doc/svgs//e5f174df537eb831a8326efc975aa4b4.svg?invert_in_darkmode" align=middle width=99.190245pt height=29.12679pt/>.
We use MFCC as our features; so <img src="https://rawgit.com/mozilla/DeepSpeech/issue313/doc/svgs//a16a13264cbafca76e1829d3e0765a69.svg?invert_in_darkmode" align=middle width=24.947835pt height=34.27314pt/> denotes the <img src="https://rawgit.com/mozilla/DeepSpeech/issue313/doc/svgs//2ec6e630f199f589a2402fdf3e0289d5.svg?invert_in_darkmode" align=middle width=8.2397205pt height=14.10255pt/>-th MFCC feature
in the audio frame at time <img src="https://rawgit.com/mozilla/DeepSpeech/issue313/doc/svgs//4f4f4e395762a3af4575de74c019ebb5.svg?invert_in_darkmode" align=middle width=5.913963pt height=20.1465pt/>. The goal of our BRNN is to convert an input
sequence <img src="https://rawgit.com/mozilla/DeepSpeech/issue313/doc/svgs//332cc365a4987aacce0ead01b8bdcc0b.svg?invert_in_darkmode" align=middle width=9.359955pt height=14.10255pt/> into a sequence of character probabilities for the transcription
<img src="https://rawgit.com/mozilla/DeepSpeech/issue313/doc/svgs//deceeaf6940a8c7a5a02373728002b0f.svg?invert_in_darkmode" align=middle width=8.61696pt height=14.10255pt/>, with <img src="https://rawgit.com/mozilla/DeepSpeech/issue313/doc/svgs//f559f3f08ccdab2e684d621045c067e7.svg?invert_in_darkmode" align=middle width=94.37901pt height=24.56553pt/>,
where <img src="https://rawgit.com/mozilla/DeepSpeech/issue313/doc/svgs//e3955e81b908e83f304c8cb51fa01cdf.svg?invert_in_darkmode" align=middle width=303.646695pt height=24.56553pt/>.
(The significance of <img src="https://rawgit.com/mozilla/DeepSpeech/issue313/doc/svgs//f99b9a2c90cd48431a7372de2cf47523.svg?invert_in_darkmode" align=middle width=39.76665pt height=22.74591pt/> will be explained below.)

Our BRNN model is composed of <img src="https://rawgit.com/mozilla/DeepSpeech/issue313/doc/svgs//9612eecfec9dadf1a81d296bd2473777.svg?invert_in_darkmode" align=middle width=8.188554pt height=21.10812pt/> layers of hidden units.
For an input <img src="https://rawgit.com/mozilla/DeepSpeech/issue313/doc/svgs//332cc365a4987aacce0ead01b8bdcc0b.svg?invert_in_darkmode" align=middle width=9.359955pt height=14.10255pt/>, the hidden units at layer <img src="https://rawgit.com/mozilla/DeepSpeech/issue313/doc/svgs//2f2322dff5bde89c37bcae4116fe20a8.svg?invert_in_darkmode" align=middle width=5.2088685pt height=22.74591pt/> are denoted <img src="https://rawgit.com/mozilla/DeepSpeech/issue313/doc/svgs//977354073f08eeac0f95545d71c2b77a.svg?invert_in_darkmode" align=middle width=23.879625pt height=29.12679pt/> with the
convention that <img src="https://rawgit.com/mozilla/DeepSpeech/issue313/doc/svgs//4187f3f834c6a9c9221fd863bd0da600.svg?invert_in_darkmode" align=middle width=26.199525pt height=29.12679pt/> is the input. The first three layers are not recurrent.
For the first layer, at each time <img src="https://rawgit.com/mozilla/DeepSpeech/issue313/doc/svgs//4f4f4e395762a3af4575de74c019ebb5.svg?invert_in_darkmode" align=middle width=5.913963pt height=20.1465pt/>, the output depends on the MFCC frame
<img src="https://rawgit.com/mozilla/DeepSpeech/issue313/doc/svgs//23776aad854f2d33e83e4f4cad44e1b9.svg?invert_in_darkmode" align=middle width=14.30715pt height=14.10255pt/> along with a context of <img src="https://rawgit.com/mozilla/DeepSpeech/issue313/doc/svgs//9b325b9e31e85137d1de765f43c0f8bc.svg?invert_in_darkmode" align=middle width=12.876435pt height=22.38192pt/> frames on each side.
(We typically use <img src="https://rawgit.com/mozilla/DeepSpeech/issue313/doc/svgs//3b3d517eaf7a36186d0f2c9bf921e14f.svg?invert_in_darkmode" align=middle width=88.45122pt height=24.56553pt/> for our experiments.)
The remaining non-recurrent layers operate on independent data for each time step.
Thus, for each time <img src="https://rawgit.com/mozilla/DeepSpeech/issue313/doc/svgs//4f4f4e395762a3af4575de74c019ebb5.svg?invert_in_darkmode" align=middle width=5.913963pt height=20.1465pt/>, the first <img src="https://rawgit.com/mozilla/DeepSpeech/issue313/doc/svgs//5dc642f297e291cfdde8982599601d7e.svg?invert_in_darkmode" align=middle width=8.188554pt height=21.10812pt/> layers are computed by:

<p align="center"><img src="https://rawgit.com/mozilla/DeepSpeech/issue313/doc/svgs//f1441afd023f38a19efe100013abeb0e.svg?invert_in_darkmode" align=middle width=184.80825pt height=21.23088pt/></p>

where <img src="https://rawgit.com/mozilla/DeepSpeech/issue313/doc/svgs//621ab064e859e3477fc9c023ef25de33.svg?invert_in_darkmode" align=middle width=189.575595pt height=24.56553pt/> is a clipped rectified-linear (ReLu)
activation function and <img src="https://rawgit.com/mozilla/DeepSpeech/issue313/doc/svgs//e7e30f89a8c77e163e112c244ba28bf2.svg?invert_in_darkmode" align=middle width=32.19414pt height=29.12679pt/>, <img src="https://rawgit.com/mozilla/DeepSpeech/issue313/doc/svgs//0b8618b400785e741ea027ec23718169.svg?invert_in_darkmode" align=middle width=21.47244pt height=29.12679pt/> are the weight matrix and bias
parameters for layer <img src="https://rawgit.com/mozilla/DeepSpeech/issue313/doc/svgs//2f2322dff5bde89c37bcae4116fe20a8.svg?invert_in_darkmode" align=middle width=5.2088685pt height=22.74591pt/>. The fourth layer is a bidirectional recurrent
layer[[1](http://www.di.ufpe.br/~fnj/RNA/bibliografia/BRNN.pdf)].
This layer includes two sets of hidden units: a set with forward recurrence,
<img src="https://rawgit.com/mozilla/DeepSpeech/issue313/doc/svgs//c51574e4f666a0701fd9e09d47458ba5.svg?invert_in_darkmode" align=middle width=27.347265pt height=29.12679pt/>, and a set with backward recurrence <img src="https://rawgit.com/mozilla/DeepSpeech/issue313/doc/svgs//b05e38531483454a1d5ee778ad02b2f5.svg?invert_in_darkmode" align=middle width=25.430625pt height=29.12679pt/>:

<p align="center"><img src="https://rawgit.com/mozilla/DeepSpeech/issue313/doc/svgs//a36ac2d62f39ae1a8b5ef15a33913cbe.svg?invert_in_darkmode" align=middle width=267.0822pt height=22.88484pt/></p>
<p align="center"><img src="https://rawgit.com/mozilla/DeepSpeech/issue313/doc/svgs//9e58ffe596dc1893dfbfad2efcf4d2a4.svg?invert_in_darkmode" align=middle width=263.0628pt height=22.88484pt/></p>

Note that <img src="https://rawgit.com/mozilla/DeepSpeech/issue313/doc/svgs//c51574e4f666a0701fd9e09d47458ba5.svg?invert_in_darkmode" align=middle width=27.347265pt height=29.12679pt/> must be computed sequentially from <img src="https://rawgit.com/mozilla/DeepSpeech/issue313/doc/svgs//a762a11a67cb9ba897faa6c3cacedbbc.svg?invert_in_darkmode" align=middle width=35.97231pt height=21.10812pt/> to <img src="https://rawgit.com/mozilla/DeepSpeech/issue313/doc/svgs//301d7bc34a1d89cb52b64ce4cf16322e.svg?invert_in_darkmode" align=middle width=54.50643pt height=29.12679pt/>
for the <img src="https://rawgit.com/mozilla/DeepSpeech/issue313/doc/svgs//77a3b857d53fb44e33b53e4c8b68351a.svg?invert_in_darkmode" align=middle width=5.642109pt height=21.60213pt/>-th utterance, while the units <img src="https://rawgit.com/mozilla/DeepSpeech/issue313/doc/svgs//b05e38531483454a1d5ee778ad02b2f5.svg?invert_in_darkmode" align=middle width=25.430625pt height=29.12679pt/> must be computed
sequentially in reverse from <img src="https://rawgit.com/mozilla/DeepSpeech/issue313/doc/svgs//301d7bc34a1d89cb52b64ce4cf16322e.svg?invert_in_darkmode" align=middle width=54.50643pt height=29.12679pt/> to <img src="https://rawgit.com/mozilla/DeepSpeech/issue313/doc/svgs//a762a11a67cb9ba897faa6c3cacedbbc.svg?invert_in_darkmode" align=middle width=35.97231pt height=21.10812pt/>.

The fifth (non-recurrent) layer takes both the forward and backward units as inputs

<p align="center"><img src="https://rawgit.com/mozilla/DeepSpeech/issue313/doc/svgs//5b71e9714fd57cbed1518f54d9674924.svg?invert_in_darkmode" align=middle width=177.2958pt height=19.479405pt/></p>

where <img src="https://rawgit.com/mozilla/DeepSpeech/issue313/doc/svgs//5544c9e9f0c04f486d43b9a513f821f4.svg?invert_in_darkmode" align=middle width=122.65968pt height=29.12679pt/>. The output layer are standard logits that
correspond to the predicted character probabilities for each time slice <img src="https://rawgit.com/mozilla/DeepSpeech/issue313/doc/svgs//4f4f4e395762a3af4575de74c019ebb5.svg?invert_in_darkmode" align=middle width=5.913963pt height=20.1465pt/> and
character <img src="https://rawgit.com/mozilla/DeepSpeech/issue313/doc/svgs//63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode" align=middle width=9.041505pt height=22.74591pt/> in the alphabet:

<p align="center"><img src="https://rawgit.com/mozilla/DeepSpeech/issue313/doc/svgs//7ba8b2b4fec039363453efb55129fc6b.svg?invert_in_darkmode" align=middle width=222.98595pt height=24.324465pt/></p>

Here <img src="https://rawgit.com/mozilla/DeepSpeech/issue313/doc/svgs//99568e8e69332dfe7e459b1be0064905.svg?invert_in_darkmode" align=middle width=23.792175pt height=34.27314pt/> denotes the <img src="https://rawgit.com/mozilla/DeepSpeech/issue313/doc/svgs//63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode" align=middle width=9.041505pt height=22.74591pt/>-th bias and <img src="https://rawgit.com/mozilla/DeepSpeech/issue313/doc/svgs//616b07a8d9f2b9a6cb5ebab869587fab.svg?invert_in_darkmode" align=middle width=82.459575pt height=34.27314pt/> the <img src="https://rawgit.com/mozilla/DeepSpeech/issue313/doc/svgs//63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode" align=middle width=9.041505pt height=22.74591pt/>-th
element of the matrix product.

Once we have computed a prediction for <img src="https://rawgit.com/mozilla/DeepSpeech/issue313/doc/svgs//e04f274d8b6ee9ec252857546b8b73b5.svg?invert_in_darkmode" align=middle width=24.105015pt height=22.74591pt/>, we compute the CTC loss
[[2]](http://www.cs.toronto.edu/~graves/preprint.pdf) <img src="https://rawgit.com/mozilla/DeepSpeech/issue313/doc/svgs//b97650f5d17fe16c62e9b16a11767663.svg?invert_in_darkmode" align=middle width=45.906795pt height=31.42161pt/>
to measure the error in prediction. During training, we can evaluate the gradient
<img src="https://rawgit.com/mozilla/DeepSpeech/issue313/doc/svgs//1e46e5cdcab1a717717603eb80c5061b.svg?invert_in_darkmode" align=middle width=59.554275pt height=31.42161pt/> with respect to the network outputs given the
ground-truth character sequence <img src="https://rawgit.com/mozilla/DeepSpeech/issue313/doc/svgs//deceeaf6940a8c7a5a02373728002b0f.svg?invert_in_darkmode" align=middle width=8.61696pt height=14.10255pt/>. From this point, computing the gradient
with respect to all of the model parameters may be done via back-propagation
through the rest of the network. We use the Adam method for training
[[3](http://arxiv.org/abs/1412.6980)].

The complete BRNN model is illustrated in the figure below.

![DeepSpeech BRNN](../images/rnn_fig-624x548.png)
