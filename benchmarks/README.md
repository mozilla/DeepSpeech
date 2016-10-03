Benchmarking CTC vs WarpCTC
===========================

We rely on Tensorflow's CTC imlementation, and WarpCTC implementation from
Baidu picked and updated from Hibbert-pku[1].

Construction of the sets:
-------------------------
We used a single audio sample, LDC93S1, repeated N times in a batch used for
training. We train (forcing overfit) this network and validate it against the
same set. Each sample is a (292,13) matrix.

The GTX970 is a desktop GPU, with 4GiB of memory. The TITAN X is a more powerful
12GiB memory-backed GPU.

# GTX970 vs TITAN X:
How to reproduce is documented in ``gpus.sh``.

The goal of this part was to assert the impact of the GPU itself:
 - computation time
 - numerical stability

In this case, we made use of the same batch size (350, that was around the
maximum before OOM would happen on the GTX970).

## Loss on GTX970 vs TITAN X
![GTX970 loss](time/warpctc_loss_gtx970_350.png?raw=true "Loss on GTX970" =100x) ![TITAN X loss](time/warpctc_loss_titanx_350.png?raw=true "Loss on TITAN X" =100x)

## Validation error on GTX970 vs TITAN X
![GTX970 validation error](time/warpctc_valerr_gtx970_350.png?raw=true "Validation error on GTX970" =100x) ![TITAN X validation error](time/warpctc_valerr_titanx_350.png?raw=true "Validation error on TITAN X" =100x)

The values of both loss evolution and validation error do behave very closely
on both GPUs, and only the standard deviation is shown to be smaller when
running on the TITAN X GPU. Execution time is, non surprisingly, in favor of
the TITAN X too.

# CTC vs WarpCTC

Tensorflow's implementation of CTC is designed to run on the CPU, but the whole
training of the network can leverage some use of the GPU.

Execution time was measured over the whole training period, wrapping around the
``session.run()`` calls, including the validation set. This way, we do compare
the whole training of the network with just making the loss function changing.

Given the existence of randomness around the process of training the network, we
made sure to:
 - run the training several times to compute mean and deviation values
 - train the model with enough iterations to make sure it was converging in
   similar ways

We limited the study to running the model 10 times, over 800 training epochs.

## Loss evolution:

![CTC loss](time/ctc_loss_titanx_1400.png?raw=true "CTC Loss" =100x) ![WarpCTC loss](time/warpctc_loss_titanx_1300.png?raw=true "WarpCTC Loss" =100x)

## Validation error

![CTC validation error](time/ctc_valerr_titanx_1400.png?raw=true "CTC validation error" =100x) ![WarpCTC validation error](time/warpctc_valerr_titanx_1300.png?raw=true "WarpCTC validation error" =100x)

## Execution time

This is being documented in the plots above. Execution time is really completely
stable on the WarpCTC implementation, with low deviation from the mean value.

Looking at the raw values, we can see that the execution time of WarpCTC on a
smaller (1300 elements) batch is still slightly higher than CTC with 1400
elements in the batch. However, looking at the loss and validation error, we can
spot a lot of missing values because loss returned an "inf" value on the
computation. This is inconclusive regarding performances so far.

## What runs where

Using Tensorflow's allocation logging, we can track what has been executed where
as documented in ``ctc_titanx_1000_loggpu.txt`` for the CTC implementation and
``warpctc_titanx_1000_loggpu.txt`` for WarpCTC.

From that first file, for CTC, we can extract that those operations runs on CPU:
```
CTCGreedyDecoder: /job:localhost/replica:0/task:0/cpu:0
edit_distance: /job:localhost/replica:0/task:0/cpu:0
Rank: /job:localhost/replica:0/task:0/cpu:0
CTCLoss: /job:localhost/replica:0/task:0/cpu:0
Placeholder_9: /job:localhost/replica:0/task:0/cpu:0
```

Thus we can just check if those ops runs on GPU or CPU within the WarpCTC log:
``for e in $(grep "cpu:0" ctc_titanx_1000_loggpu.txt | grep -v '^. ' |sort | uniq |cut -d':' -f1); do echo -n "CTC $e => "; grep "$e:" warpctc_titanx_1000_loggpu.txt | grep -v '^. ' | sort | uniq; done;``

Which gives us:
```
CTC CTCGreedyDecoder => CTCGreedyDecoder: /job:localhost/replica:0/task:0/cpu:0
CTC CTCLoss => WarpCTCLoss: /job:localhost/replica:0/task:0/gpu:0
CTC edit_distance => edit_distance: /job:localhost/replica:0/task:0/cpu:0
CTC Placeholder_9 => Placeholder_9: /job:localhost/replica:0/task:0/cpu:0
CTC Rank => Rank: /job:localhost/replica:0/task:0/cpu:0
```

We can see that the only difference at runtime is: CTCLoss operation runs on
CPU when WarpCTCLoss runs on GPU.

# Batch size:
How to reproduce is documented in ``batch.sh``.

Below is the plot of the data. Batch size was changed, ranging from 1 to 1600,
in steps of 10 except between 1 and 64, where the square value was used (1, 2,
4, 8, ..., 64). Above a batch size of 1430, execution starts to fail because of
the GPU running out of memory.

For each batch size value, we did three runs of 200 epochs. The resulting
execution time is being taken for the mean/stddev value. We know WarpCTC
execution time is more stable so we can live with just three runs.

Scaling is shown to be linear.

![Batch size influence](batch/warpctc_titanx_batch_1-1600.png?raw=true "Batch size influence" =100x)

[1]: https://github.com/Hibbert-pku/tensorflow/commit/b5a569163921e6c57d86ab57301d23a2948c3729
