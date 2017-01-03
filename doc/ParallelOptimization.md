# Parallel Optimization

This is how we implement optimization of the DeepSpeech model across GPU's on a
single host. Parallel optimization can take on various forms. For example
one can use asynchronous updates of the model, synchronous updates of the model,
or some combination of the two.

## Asynchronous Parallel Optimization

In asynchronous parallel optimization, for example, one places the model
initially in CPU memory. Then each of the <img src="https://rawgit.com/mozilla/DeepSpeech/issue313/doc/svgs//5201385589993766eea584cd3aa6fa13.svg?invert_in_darkmode" align=middle width=12.876435pt height=22.38192pt/> GPU's obtains a mini-batch of data
along with the current model parameters. Using this mini-batch each GPU then
computes the gradients for all model parameters and sends these gradients back
to the CPU when the GPU is done with its mini-batch. The CPU then asynchronously
updates the model parameters whenever it recieves a set of gradients from a GPU.

Asynchronous parallel optimization has several advantages and several
disadvantages. One large advantage is throughput. No GPU will every be waiting
idle. When a GPU is done processing a mini-batch, it can immediately obtain the
next mini-batch to process. It never has to wait on other GPU's to finish their
mini-batch. However, this means that the model updates will also be asynchronous
which can have problems.

For example, one may have model parameters <img src="https://rawgit.com/mozilla/DeepSpeech/issue313/doc/svgs//84c95f91a742c9ceb460a83f9b5090bf.svg?invert_in_darkmode" align=middle width=17.74179pt height=22.38192pt/> on the CPU and send mini-batch
<img src="https://rawgit.com/mozilla/DeepSpeech/issue313/doc/svgs//55a049b8f161ae7cfeb0197d75aff967.svg?invert_in_darkmode" align=middle width=9.83004pt height=14.10255pt/> to GPU 1 and send mini-batch <img src="https://rawgit.com/mozilla/DeepSpeech/issue313/doc/svgs//3f18d8f60c110e865571bba5ba67dcc6.svg?invert_in_darkmode" align=middle width=38.062035pt height=21.10812pt/> to GPU 2. As processing is asynchronous,
GPU 2 may finish before GPU 1 and thus update the CPU's model parameters <img src="https://rawgit.com/mozilla/DeepSpeech/issue313/doc/svgs//84c95f91a742c9ceb460a83f9b5090bf.svg?invert_in_darkmode" align=middle width=17.74179pt height=22.38192pt/>
with its gradients <img src="https://rawgit.com/mozilla/DeepSpeech/issue313/doc/svgs//d945d0d19d93957738ab47285585af55.svg?invert_in_darkmode" align=middle width=85.19445pt height=24.56553pt/>, where the subscript <img src="https://rawgit.com/mozilla/DeepSpeech/issue313/doc/svgs//3f18d8f60c110e865571bba5ba67dcc6.svg?invert_in_darkmode" align=middle width=38.062035pt height=21.10812pt/> identifies the
mini-batch and the argument <img src="https://rawgit.com/mozilla/DeepSpeech/issue313/doc/svgs//84c95f91a742c9ceb460a83f9b5090bf.svg?invert_in_darkmode" align=middle width=17.74179pt height=22.38192pt/> the location at which the gradient was evaluated.
This results in the new model parameters

<p align="center"><img src="https://rawgit.com/mozilla/DeepSpeech/issue313/doc/svgs//821c06a38f428b87e586f7b26800c3af.svg?invert_in_darkmode" align=middle width=127.537245pt height=16.376943pt/></p>

Next GPU 1 could finish with its mini-batch and update the parameters to

<p align="center"><img src="https://rawgit.com/mozilla/DeepSpeech/issue313/doc/svgs//9f326591601cc1a07f0590bbed3768d8.svg?invert_in_darkmode" align=middle width=216.13185pt height=16.376943pt/></p>

The problem with this is that <img src="https://rawgit.com/mozilla/DeepSpeech/issue313/doc/svgs//f8a131f8ae97558cd5a81df2db955b0f.svg?invert_in_darkmode" align=middle width=68.55057pt height=24.56553pt/> is evaluated at <img src="https://rawgit.com/mozilla/DeepSpeech/issue313/doc/svgs//84c95f91a742c9ceb460a83f9b5090bf.svg?invert_in_darkmode" align=middle width=17.74179pt height=22.38192pt/> and not
<img src="https://rawgit.com/mozilla/DeepSpeech/issue313/doc/svgs//866ca4413312fc41c943f3c4fbca73c1.svg?invert_in_darkmode" align=middle width=122.988195pt height=24.56553pt/>. Hence, the direction of the gradient <img src="https://rawgit.com/mozilla/DeepSpeech/issue313/doc/svgs//f8a131f8ae97558cd5a81df2db955b0f.svg?invert_in_darkmode" align=middle width=68.55057pt height=24.56553pt/>
is slightly incorrect as it is evaluated at the wrong location. This can be
counteracted through synchronous updates of model, but this is also problematic.

## Synchronous Optimization

Synchronous optimization solves the problem we saw above. In synchronous
optimization, one places the model initially in CPU memory. Then one of the <img src="https://rawgit.com/mozilla/DeepSpeech/issue313/doc/svgs//5201385589993766eea584cd3aa6fa13.svg?invert_in_darkmode" align=middle width=12.876435pt height=22.38192pt/>
GPU's is given a mini-batch of data along with the current model parameters.
Using the mini-batch the GPU computes the gradients for all model parameters and
sends the gradients back to the CPU. The CPU then updates the model parameters
and starts the process of sending out the next mini-batch.

As on can readily see, synchronous optimization does not have the problem we
found in the last section, that of incorrect gradients. However, synchronous
optimization can only make use of a single GPU at a time. So, when we have a
multi-GPU setup, <img src="https://rawgit.com/mozilla/DeepSpeech/issue313/doc/svgs//f1484f55b33675c60d5cc8a5689ba737.svg?invert_in_darkmode" align=middle width=42.934815pt height=22.38192pt/>, all but one of the GPU's will remain idle, which is
unacceptable. However, there is a third alternative which is combines the
advantages of asynchronous and synchronous optimization.

## Hybrid Parallel Optimization

Hybrid parallel optimization combines most of the benifits of asynchronous and
synchronous optimization. It allows for multiple GPU's to be used, but does not
suffer from the incorrect gradient problem exhibited by asynchronous
optimization.

In hybrid parallel optimization one places the model initially in CPU memory.
Then, as in asynchronous optimization, each of the <img src="https://rawgit.com/mozilla/DeepSpeech/issue313/doc/svgs//5201385589993766eea584cd3aa6fa13.svg?invert_in_darkmode" align=middle width=12.876435pt height=22.38192pt/> GPU'S obtains a
mini-batch of data along with the current model parameters. Using the mini-batch
each of the GPU's then computes the gradients for all model parameters and sends
these gradients back to the CPU. Now, in contrast to asynchronous optimization,
the CPU waits until each GPU is finished with its mini-batch then takes the mean
of all the gradients from the <img src="https://rawgit.com/mozilla/DeepSpeech/issue313/doc/svgs//5201385589993766eea584cd3aa6fa13.svg?invert_in_darkmode" align=middle width=12.876435pt height=22.38192pt/> GPU's and updates the model with this mean
gradient.

<img src="images/Parallelism.png" alt="LSTM" width="600">

Hybrid parallel optimization has several advantages and few disadvantages. As in
asynchronous parallel optimization, hybrid parallel optimization allows for one
to use multiple GPU's in parallel. Furthermore, unlike asynchronous parallel
optimization, the incorrect gradient problem is not present here. In fact,
hybrid parallel optimization performs as if one is working with a single
mini-batch which is <img src="https://rawgit.com/mozilla/DeepSpeech/issue313/doc/svgs//5201385589993766eea584cd3aa6fa13.svg?invert_in_darkmode" align=middle width=12.876435pt height=22.38192pt/> times the size of a mini-batch handled by a single GPU.
Hoewever, hybrid parallel optimization is not perfect. If one GPU is slower than
all the others in completing its mini-batch, all other GPU's will have to sit
idle until this straggler finishes with its mini-batch. This hurts throughput.
But, if all GPU'S are of the same make and model, this problem should be
minimized.

So, relatively speaking, hybrid parallel optimization seems the have more
advantages and fewer disadvantages as compared to both asynchronous and
synchronous optimization. So, we will, for our work, use this hybrid model.

## Adam Optimization

In constrast to
[Deep Speech: Scaling up end-to-end speech recognition](http://arxiv.org/abs/1412.5567),
in which  
[Nesterov’s Accelerated Gradient Descent](www.cs.toronto.edu/~fritz/absps/momentum.pdf)
was used, we will use the Adam method for optimization
[[3](http://arxiv.org/abs/1412.6980)],
because, generally, it requires less fine-tuning.
