
# Introduction

In this notebook we will reproduce the results of
[Deep Speech: Scaling up end-to-end speech recognition](http://arxiv.org/abs/1412.5567).
The core of the system is a bidirectional recurrent neural network (BRNN)
trained to ingest speech spectrograms and generate English text transcriptions.

Let a single utterance $x$ and label $y$ be sampled from a training set
$S = \{(x^{(1)}, y^{(1)}), (x^{(2)}, y^{(2)}), . . .\}$.
Each utterance, $x^{(i)}$ is a time-series of length $T^{(i)}$
where every time-slice is a vector of audio features,
$x^{(i)}_t$ where $t=1,\ldots,T^{(i)}$.
We use MFCC as our features; so $x^{(i)}_{t,p}$ denotes the $p$-th MFCC feature
in the audio frame at time $t$. The goal of our BRNN is to convert an input
sequence $x$ into a sequence of character probabilities for the transcription
$y$, with $\hat{y}_t =\mathbb{P}(c_t \mid x)$,
where $c_t \in \{a,b,c, . . . , z, space, apostrophe, blank\}$.
(The significance of $blank$ will be explained below.)

Our BRNN model is composed of $5$ layers of hidden units.
For an input $x$, the hidden units at layer $l$ are denoted $h^{(l)}$ with the
convention that $h^{(0)}$ is the input. The first three layers are not recurrent.
For the first layer, at each time $t$, the output depends on the MFCC frame
$x_t$ along with a context of $C$ frames on each side.
(We typically use $C \in \{5, 7, 9\}$ for our experiments.)
The remaining non-recurrent layers operate on independent data for each time step.
Thus, for each time $t$, the first $3$ layers are computed by:

$$h^{(l)}_t = g(W^{(l)} h^{(l-1)}_t + b^{(l)})$$

where $g(z) = \min\{\max\{0, z\}, 20\}$ is a clipped rectified-linear (ReLu)
activation function and $W^{(l)}$, $b^{(l)}$ are the weight matrix and bias
parameters for layer $l$. The fourth layer is a bidirectional recurrent
layer[[1](http://www.di.ufpe.br/~fnj/RNA/bibliografia/BRNN.pdf)].
This layer includes two sets of hidden units: a set with forward recurrence,
$h^{(f)}$, and a set with backward recurrence $h^{(b)}$:

$$h^{(f)}_t = g(W^{(4)} h^{(3)}_t + W^{(f)}_r h^{(f)}_{t-1} + b^{(4)})$$
$$h^{(b)}_t = g(W^{(4)} h^{(3)}_t + W^{(b)}_r h^{(b)}_{t+1} + b^{(4)})$$

Note that $h^{(f)}$ must be computed sequentially from $t = 1$ to $t = T^{(i)}$
for the $i$-th utterance, while the units $h^{(b)}$ must be computed
sequentially in reverse from $t = T^{(i)}$ to $t = 1$.

The fifth (non-recurrent) layer takes both the forward and backward units as inputs

$$h^{(5)} = g(W^{(5)} h^{(4)} + b^{(5)})$$

where $h^{(4)} = h^{(f)} + h^{(b)}$. The output layer are standard logits that
correspond to the predicted character probabilities for each time slice $t$ and
character $k$ in the alphabet:

$$h^{(6)}_{t,k} = \hat{y}_{t,k} = (W^{(6)} h^{(5)}_t)_k + b^{(6)}_k$$

Here $b^{(6)}_k$ denotes the $k$-th bias and $(W^{(6)} h^{(5)}_t)_k$ the $k$-th
element of the matrix product.

Once we have computed a prediction for $\hat{y}_{t,k}$, we compute the CTC loss
[[2]](http://www.cs.toronto.edu/~graves/preprint.pdf) $\cal{L}(\hat{y}, y)$
to measure the error in prediction. During training, we can evaluate the gradient
$\nabla \cal{L}(\hat{y}, y)$ with respect to the network outputs given the
ground-truth character sequence $y$. From this point, computing the gradient
with respect to all of the model parameters may be done via back-propagation
through the rest of the network. We use the Adam method for training
[[3](http://arxiv.org/abs/1412.6980)].

The complete BRNN model is illustrated in the figure below.

![DeepSpeech BRNN](../images/rnn_fig-624x548.png)
