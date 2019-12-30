# Finetune Language Model Parameters: Alpha, Beta
This topic is to explain how did `finetune_lm_params.py` fit `alpha` and `beta`

## Inspiration
According to mesh plot result of [Paddle/DeepSpeech-Grid-Search](https://github.com/PaddlePaddle/DeepSpeech/raw/develop/docs/images/tuning_error_surface.png), I assume `BatchWER v.s. Alpha` is `ideal valley parabolic curve`, i.e. which fits this formula below:

Idea Valley Parabola Formula
===
- Assume there exists a `lowest WER` satisfies the formula: ![A.1](https://latex.codecogs.com/gif.latex?WER_{batch}=A_0(\\alpha-\\alpha_{best})^2+A_1 "A.1"),  `(A.1)`
- Where ![](https://latex.codecogs.com/gif.latex?A_0>0) and ![](https://latex.codecogs.com/gif.latex?A_1>0)
- If we set ![](https://latex.codecogs.com/gif.latex?\\alpha=\\alpha_{best}), the `WER` should be lowest: ![](https://latex.codecogs.com/gif.latex?WER_{lowest}=A_1), so the target is to find out the constant ![](https://latex.codecogs.com/gif.latex?\\alpha_{best})
- To solve the `best alpha`, I decompose the formula, and change the coefficient as below:
- ![A.2](https://latex.codecogs.com/gif.latex?WER_{batch}=A_0\\alpha^2-2A_0\\alpha_{best}\\alpha+A_1\\alpha_{best}^2+A_1 "A.2"),  `(A.2)`

Polynomial Idea Valley Parabola Formula
===
- Let's tidy the `(A.2)` coefficients as below:
- Assume there exists a `lowest WER` satisfies the formula: ![B.1](https://latex.codecogs.com/gif.latex?WER_{Batch}=B_0\\alpha^2+B_1\\alpha+B_2 "B.1"),  `(B.1)`
- Where ![](https://latex.codecogs.com/gif.latex?B_0>0,B_1<0,B_2>0)
- Compare to `(A.1)`: ![](https://latex.codecogs.com/gif.latex?B_0=A_0) and ![](https://latex.codecogs.com/gif.latex?B_1=-2A_0\\alpha_{best})
- So, with formula `(B.1)`, we can derive: ![](https://latex.codecogs.com/gif.latex?\\alpha_{best}=-\\frac{B_1}{B_0})
- To make computer calculations easier and consume the least amount of computing power, `Statistics` + `Linear Algebra` should be a good choice, so let's transform `(B.1)` as `matrix` form and introduce `statistics` assumption.


Valley Parabola Formula in Matrix Form
===
- Assume testing one sampling audio with ![](https://latex.codecogs.com/gif.latex?\\alpha_n) by `test()` function can get ![](https://latex.codecogs.com/gif.latex?output_n), where `output` can be considered as `wer`/`cer`/`word_distance`/`char_distance`
- Then if do an experiment on testing sampling data, with different `alpha` settings, we should get a bunch outputs: ![](https://latex.codecogs.com/gif.latex?\\textbf{Y}=\begin{bmatrix}output_0&output_1&\.\.\.&output_n\end{bmatrix})
- To solve the best parameter issue by experiment data, we should introduce statistics method
- Assume we have found a set of `BEST` coefficients: ![](https://latex.codecogs.com/gif.latex?\\hat{\\textbf{C}}=\begin{bmatrix}\\hat{C_0}&\\hat{C_1}&\\hat{C_2}\end{bmatrix}), which fits our formula `(B.2)` to derive outputs as ![](https://latex.codecogs.com/gif.latex?\\hat{\\textbf{Y}}=\begin{bmatrix}\\hat{output_0}&\\hat{output_1}&\.\.\.&\\hat{output_n}\end{bmatrix}), the matrix form should be:<br/>
![](https://latex.codecogs.com/gif.latex?\\textbf{X}\\hat{\\textbf{C}^T}=\\hat{\\textbf{Y}^T}), i.e. ![](https://latex.codecogs.com/gif.latex?\\begin{bmatrix}\\alpha_0^2&\\alpha_1&1\\\\\alpha_1^2&\\alpha_2&1\\\\.\.\.\\\\\alpha_n^2&\\alpha_n&1\end{bmatrix}\\begin{bmatrix}\\hat{C0}\\\\\hat{C1}\\\\\hat{C2}\end{bmatrix}=\begin{bmatrix}\\hat{output_0}\\\\\hat{output_1}\\\\.\.\.\\\\\hat{output_n}\end{bmatrix} "C.1"), `(C.1)`
- And the `BEST` coefficient ![](https://latex.codecogs.com/gif.latex?\\hat{\\textbf{C}}) should keep ![](https://latex.codecogs.com/gif.latex?(\\textbf{Y}-\\hat{\\textbf{Y}})^2) to be minimum.
- If we founded ![](https://latex.codecogs.com/gif.latex?\\hat{\\textbf{C}}), we can derive ![](https://latex.codecogs.com/gif.latex?\\hat{\\alpha_{best}}=-\\frac{\\hat{C_1}}{\\hat{C_0}}), which is the `best alpha` we have estimated.

Solve Formula `(C.1)`
===
- Because we can never get absolute solution of `(B.1)`, unless iterate every float value on every single audio, so we use instead `statistics` to solve `(C.1)` closest to `(B.1)`
- I use `Pseudo Inverse` directly to solve ![](https://latex.codecogs.com/gif.latex?\\hat{\\textbf{C}^T}=\\textbf{X}^\\dagger\\hat{\\textbf{Y}}), in numpy, the `Pseudo Inverse` is `numpy.linalg.pinv`

Solve Best LM Paramters
===
- Now, we have `pseudo inverse tool` to fit every parabola curve, the rest job is coding then
- About how to scan and sampling the data is not such absolutely