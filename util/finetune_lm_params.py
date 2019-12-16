#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

# Make sure we can import stuff from util/
# This script needs to be run from the root of the DeepSpeech repository
import pandas as pd
import numpy as np
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

def finetune_parabola(param_result_list, param_min, param_max):
    # input: <list>
    # param_result_list = [{
    #     "param": ...,
    #     "result": ...,
    # }, {
    #     "param": ...,
    #     "result": ...,
    # }]

    # assume:
    #   result0 = C0 * param0^2 + C1 * param0 + C2
    #   result1 = C0 * param1^2 + C1 * param1 + C2
    #   ...
    # X = [[param0^2, param0, 1], [param1^2, param1, 1], ...]
    # Y = [[result0], [result1], ...]
    # A = [[C0], [C1], [C2]]
    # XA = Y
    # if C0 > 0 => best_alpha = - C1 / 2 / C0
    # if C0 < 0 => best_alpha = X[argmin(wer)]
    mat_x = []
    mat_y = []
    for param_result in param_result_list:
        p = param_result['param']
        mat_x.append([p**2, p, 1.0])
        mat_y.append([param_result['result']])
    mat_a = np.matmul(np.linalg.pinv(mat_x), mat_y)
    c0 = mat_a[0, 0]
    c1 = mat_a[1, 0]
    if c0 <= 0:
        print("#### [Fit Failed] it's not a ideal parabola, so just pick a lowest parameter ####")
        return pick_lowest_param(param_result_list)

    # the parabola has minimum param
    return max(min(- c1 / 2.0 / c0, param_max), param_min)


def finetune_linear(param_result_list, param_min, param_max):
    # input: <list>
    # param_result_list = [{
    #     "param": ...,
    #     "result": ...,
    # }, {
    #     "param": ...,
    #     "result": ...,
    # }]

    # assume:
    #   result0 = C0 * param0 + C1
    #   result1 = C0 * param1 + C1
    #   ...
    # X = [[param0, 1], [param1, 1], ...]
    # Y = [[result0], [result1], ...]
    # A = [[C0], [C1]]
    # XA = Y

    mat_x = []
    mat_y = []
    for param_result in param_result_list:
        p = param_result['param']
        mat_x.append([p, 1.0])
        mat_y.append([param_result['result']])
    mat_a = np.matmul(np.linalg.pinv(mat_x), mat_y)
    c0 = mat_a[0, 0]
    return param_min if c0 > 0 else param_max


def pick_lowest_param(param_result_list):
    df = pd.DataFrame(param_result_list)
    return float(df.groupby('param').mean()['result'].idxmin())
