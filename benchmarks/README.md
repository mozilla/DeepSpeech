Benchmarking CTC vs WarpCTC
===========================

We rely on Tensorflow's CTC imlementation, and WarpCTC implementation from
Baidu picked and updated from Hibbert-pku[1].

# Construction of the sets:

We used a single audio sample, LDC93S1, repeated N times in a batch used for
training. We train (forcing overfit) this network and validate it against the
same set. Each sample is a (292,13) matrix.

# Measuring

We measured execution time (along with other parameters) from the DeepSpeech
notebook (DeepSpeech.ipynb) and recorded all that in CSV files.

# Mono-GPU

We compared CTC vs WarpCTC on a set of 128 and 512 audio samples as documented
above. Training has been pushed for 300 iterations, and this has been repeated
10 times.

We forced only using GPU #0 over the 4 available on the system. In both batch
size, the use of WarpCTC has a slight advantage in term of execution time.

## Batch size 128

Data can be found in:
 * ``ctc_titanx_nepochs300_batch128_1gpu.csv`` for CTC
 * ``warpctc_titanx_nepochs300_batch128_1gpu.csv`` for WarpCTC
```
python ../plot.py --csv warpctc_titanx_nepochs300_batch128_1gpu.csv --plot warpctc_titanx_nepochs300_batch128_1gpu.png
python ../plot.py --csv ctc_titanx_nepochs300_batch128_1gpu.csv --plot ctc_titanx_nepochs300_batch128_1gpu.png
```

![TITAN X Mono-GPU batch size 128: CTC loss](ctc_titanx_nepochs300_batch128_1gpu.png?raw=true "CTC Loss on TITAN X with one GPU and batch size 128") ![TITAN X Mono-GPU batch size 128: WarpCTC loss](warpctc_titanx_nepochs300_batch128_1gpu.png?raw=true "WarpCTC Loss on TITAN X with one GPU and batch size 128")

## Batch size 512

Data can be found in:
 * ``ctc_titanx_nepochs300_batch512_1gpu.csv`` for CTC
 * ``warpctc_titanx_nepochs300_batch512_1gpu.csv`` for WarpCTC

Plots can be regenerated:
```
python ../plot.py --csv warpctc_titanx_nepochs300_batch512_1gpu.csv --plot warpctc_titanx_nepochs300_batch512_1gpu.png
python ../plot.py --csv ctc_titanx_nepochs300_batch512_1gpu.csv --plot ctc_titanx_nepochs300_batch512_1gpu.png
```

![TITAN X Mono-GPU batch size 512: CTC loss](ctc_titanx_nepochs300_batch512_1gpu.png?raw=true "CTC Loss on TITAN X with one GPU and batch size 512") ![TITAN X Mono-GPU batch size 512: WarpCTC loss](warpctc_titanx_nepochs300_batch512_1gpu.png?raw=true "WarpCTC Loss on TITAN X with one GPU and batch size 512")

# Multi-GPU

Comparison is done the same way as exposed before, except that this time we will
make use of the 4 GPUs available on the system.

## Batch size 128

Data can be found in:
 * ``ctc_titanx_nepochs300_batch128_4gpus.csv`` for CTC
 * ``warpctc_titanx_nepochs300_batch128_4gpus.csv`` for WarpCTC
```
python ../plot.py --csv warpctc_titanx_nepochs300_batch128_4gpus.csv --plot warpctc_titanx_nepochs300_batch128_4gpus.png
python ../plot.py --csv ctc_titanx_nepochs300_batch128_4gpus.csv --plot ctc_titanx_nepochs300_batch128_4gpus.png
```

## Batch size 512

Data can be found in:
 * ``ctc_titanx_nepochs300_batch512_4gpus.csv`` for CTC
 * ``warpctc_titanx_nepochs300_batch512_4gpus.csv`` for WarpCTC

Plots can be regenerated:
```
python ../plot.py --csv warpctc_titanx_nepochs300_batch512_4gpus.csv --plot warpctc_titanx_nepochs300_batch512_4gpus.png
python ../plot.py --csv ctc_titanx_nepochs300_batch512_4gpus.csv --plot ctc_titanx_nepochs300_batch512_4gpus.png

[1]: https://github.com/Hibbert-pku/tensorflow/commit/b5a569163921e6c57d86ab57301d23a2948c3729
