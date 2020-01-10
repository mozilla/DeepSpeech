# Finetune LM Method: parabola
This topic is to explain how did `finetune_lm_params.py` fit `alpha` and `beta`

### Inspiration
According to mesh plot result of [Paddle/DeepSpeech-Grid-Search](https://github.com/PaddlePaddle/DeepSpeech/raw/develop/docs/images/tuning_error_surface.png), We assume `BatchWER v.s. Alpha` is `ideal valley parabolic curve`, which means it's `2 Degrees of Polynomial` but also with some constraints.

### 2 Degrees of Polynomial
Assume there exists a `lowest WER` satisfies the formula:

![A.1](https://latex.codecogs.com/gif.latex?WER_{batch}(\alpha)=A_0\alpha^2+A_1\alpha+A_2 "A.1"),  `(A.1)`

Where the `lowest WER` is:

![A.2](https://latex.codecogs.com/gif.latex?WER_{min}=WER_{Batch}(\alpha=\alpha_{best}) "A.2"),  `(A.2)`

And the lowest point must satisfy the formula:

![A.3](https://latex.codecogs.com/gif.latex?\\dfrac{dWER_{batch}(\alpha)}{d\alpha}|_{\alpha=\alpha_{best}}=0＝=2A_0\alpha_{best}+A_1 "A.3"),  `(A.3)`


To make sure the `best alpha` is not on the [saddle point](https://en.wikipedia.org/wiki/Saddle_point), we must constraint the formula:

![A.4](https://latex.codecogs.com/gif.latex?\\dfrac{d^2WER_{batch}(\alpha)}{d\alpha^2}=2A_0>0 "A.4"),  `(A.4)`

After arrange `(A.3)`, `(A.4)`, we got `best alpha`:

![A.5](https://latex.codecogs.com/gif.latex?\alpha_{best}=-\frac{A_1}{2A_0} "A.5"), `(A.5)`

And because alpha must > 0, so:

![](https://latex.codecogs.com/gif.latex?A_0>0,A_1<0)

To make computer calculations easier and consume the least amount of computing power, `Statistics` + `Linear Algebra` should be a good choice, so let's transform `(A.1)` as `matrix` form and introduce `statistics` method: "use a subpopulation by sampling from population to estimate polulation's features"


### Matrix Form
If we infer the j-th audio from sampling data with ![](https://latex.codecogs.com/gif.latex?\alpha_i) can get ![](https://latex.codecogs.com/gif.latex?loss_{ij}), where the `loss` can be considered as `wer`/`cer`/`word_distance`/`char_distance`, and the sample size is `m`, the amount of `alpha` is `n`.

If we mean the loss from j=1~m for each `alpha_i`, then we will get a bunch outputs:

![](https://latex.codecogs.com/gif.latex?\textbf{Y}=\begin{bmatrix}\bar{loss_0}&\bar{loss_1}&\.\.\.&\bar{loss_i}&\.\.\.&\bar{loss_n}\end{bmatrix})

Assume we have found a set of `BEST` coefficients:

![](https://latex.codecogs.com/gif.latex?\hat{\textbf{C}}=\begin{bmatrix}\hat{C_0}&\hat{C_1}&\hat{C_2}\end{bmatrix})

which fits our formula `(A.1)` to derive outputs as:

![](https://latex.codecogs.com/gif.latex?\\hat{\textbf{Y}}=\begin{bmatrix}\hat{\bar{loss_0}}&\\hat{\bar{loss_1}}&\.\.\.&\hat{\\bar{loss_i}}&\.\.\.&\hat{\bar{loss_n}}\end{bmatrix})

the matrix form should be:<br/>
![](https://latex.codecogs.com/gif.latex?\textbf{X}\\hat{\textbf{C}^T}=\\hat{\textbf{Y}^T}), i.e. ![](https://latex.codecogs.com/gif.latex?\begin{bmatrix}\alpha_0^2&\alpha_1&1\\\\\alpha_1^2&\alpha_2&1\\\\.\.\.\\\\\\alpha_n^2&\alpha_n&1\end{bmatrix}\begin{bmatrix}\\hat{C0}\\\\\hat{C1}\\\\\hat{C2}\end{bmatrix}=\begin{bmatrix}\\hat{\bar{loss_0}}\\\\\hat{\bar{loss_1}}\\\\.\.\.\\\\\hat{\bar{loss_n}}\end{bmatrix} "C.1"), `(B.1)`

And the `BEST` coefficient ![](https://latex.codecogs.com/gif.latex?\\hat{\textbf{C}}) should keep residual ![](https://latex.codecogs.com/gif.latex?(\textbf{Y}-\\hat{\textbf{Y}})^2) to be minimum.

If we find ![](https://latex.codecogs.com/gif.latex?\\hat{\textbf{C}}), we can derive ![](https://latex.codecogs.com/gif.latex?\hat{\alpha_{best}}=-\frac{\hat{C_1}}{2\hat{C_0}}), which should be the `best alpha` we estimate.

### Solve Formula `(B.1)`
We use `Pseudo Inverse` directly to solve ![](https://latex.codecogs.com/gif.latex?\hat{\textbf{C}^T}=\textbf{X}^\dagger\hat{\textbf{Y}}), in numpy, the `Pseudo Inverse` is `numpy.linalg.pinv`
The code implementation is in `finetune_lm_params.py::fit_parabola()`