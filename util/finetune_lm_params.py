"""Finetune LM Parameters package

finetune_lm(create_model, try_loading, original_samples, method) is the entrypoint.

To implement new algorithm:

1. we should define a function:
Parameters
----------
    sample_csvs: List[str]
        The list of samples csv filenames
    create_model: function
        The function of DeepSpeech.py::create_model to pass into evaluate function
    try_loading: function
        The function of DeepSpeech.py::try_loading to pass into evaluate function
    * the other parameters of the function required should just get from FLAGS
Returns
----------
    result: dict
        The dict must contain fields: 'best_alpha', 'best_beta', and the other infos you want to log

2. After function defined, register it into `VALID_METHODS`, which is defined at the bottom of this file after all functions are declared, Ex:
```
VALID_METHODS = {
    'parabola': finetune_lm_parabola,
    'random_search': finetune_random_search,
    'parabola+random_search': finetune_parabola_and_random_search,
    'custom': foo,
}
```

3. Then also register `new field`(Ex: 'custom') into `--finetune_lm_method`'s enum_values in flags.py
"""

from __future__ import absolute_import, division, print_function

from evaluate import evaluate
import pandas as pd
import numpy as np
import json
import tempfile
from util.feeding import read_csvs
from util.flags import FLAGS
from util.evaluate_tools import wer_cer_batch
from util.logging import log_info, log_warn
import tensorflow.compat.v1 as tfv1
from pprint import pprint
import os


# VALID_METHODS is defined at bottom after functions are defined
# To Add new finetune algorithms, which should implement:
#   input: sample_csvs<list>, create_model<func>, try_loading<func>
#   output: result<dict>

def validate_finetune_lm_flags():
    assert FLAGS.finetune_lm_alpha_max > FLAGS.finetune_lm_alpha_min
    assert FLAGS.finetune_lm_beta_max > FLAGS.finetune_lm_beta_min

    if FLAGS.finetune_lm_output_file:
        if os.path.exists(FLAGS.finetune_lm_output_file):
            log_warn("--finetune_lm_output_file {} already exists, which will be overwrited after LM finetune finish".format(
                FLAGS.finetune_lm_output_file
            ))


def sampling(csvs, sample_size):
    df = read_csvs(csvs)

    mod_sample_size = min(len(df), sample_size)
    if mod_sample_size < sample_size:
        log_warn('sample size: {} is out of data size, reset finetune LM sample size as {}'.format(
            sample_size, mod_sample_size))

    df = df.sample(n=mod_sample_size)

    sample_file = tempfile.NamedTemporaryFile('w')
    df.to_csv(sample_file.name)
    sample_csvs = [sample_file.name]
    return sample_csvs, sample_file, mod_sample_size


def finetune_lm(create_model, try_loading, original_samples, method='parabola'):
    validate_finetune_lm_flags()

    log_info('Enable finetune LM method: "{}"'.format(method))
    finetune_func = VALID_METHODS[method]

    sample_csvs, sample_file, sample_size = sampling(FLAGS.finetune_lm_csv_files, FLAGS.finetune_lm_sample_size)
    finetune_result = finetune_func(sample_csvs, create_model, try_loading)
    sample_file.close()

    assert 'best_alpha' in finetune_result
    assert 'best_beta' in finetune_result

    best_alpha = finetune_result['best_alpha']
    best_beta = finetune_result['best_beta']

    if FLAGS.test_files:
        tfv1.reset_default_graph()

        log_info('Final test with best alpha = {}, beta = {}'.format(
            best_alpha, best_beta
        ))
        finetune_samples = evaluate(FLAGS.test_files.split(','), create_model, try_loading, best_alpha, best_beta, 0)
        original_samples_wer, original_samples_cer = wer_cer_batch(original_samples)
        finetune_samples_wer, finetune_samples_cer = wer_cer_batch(finetune_samples)

        target_loss = FLAGS.finetune_lm_target_mean_loss
        original_mean_loss = np.mean([sample[target_loss] for sample in original_samples])
        finetune_mean_loss = np.mean([sample[target_loss] for sample in finetune_samples])

        finetune_result.update({
            'test_csv_files': FLAGS.test_files.split(','),
            'method': method,
            'sample_size': sample_size,
        })

        report = {
            'origin_test_result': {
                'alpha': FLAGS.lm_alpha,
                'beta': FLAGS.lm_beta,
                'samples_wer': original_samples_wer,
                'samples_cer': original_samples_cer,
                'target_mean_loss': original_mean_loss,
            },
            'finetune_test_result': {
                'alpha': best_alpha,
                'beta': best_beta,
                'samples_wer': finetune_samples_wer,
                'samples_cer': finetune_samples_cer,
                'target_mean_loss': finetune_mean_loss,
            },
            'finetune_parameters': finetune_result,
        }

        log_info('Finetune LM result:')
        pprint(report)
        if FLAGS.finetune_lm_output_file:
            log_info('Export finetune report at {}'.format(FLAGS.finetune_lm_output_file))
            open(FLAGS.finetune_lm_output_file, 'w', encoding='utf8').write(json.dumps(report, indent=4))


def finetune_random_search(sample_csvs, create_model, try_loading, alpha_start=None, beta_start=None):

    # shorten var
    alpha_min = FLAGS.finetune_lm_alpha_min
    alpha_max = FLAGS.finetune_lm_alpha_max
    beta_min = FLAGS.finetune_lm_beta_min
    beta_max = FLAGS.finetune_lm_beta_max
    target_loss = FLAGS.finetune_lm_target_mean_loss

    # If doesn't specify start position, use FLAGS
    best_alpha = FLAGS.lm_alpha if alpha_start is None else alpha_start
    best_beta = FLAGS.lm_beta if beta_start is None else beta_start

    # Set initial position
    alpha_start = best_alpha
    beta_start = best_beta
    tfv1.reset_default_graph()
    samples = evaluate(sample_csvs, create_model, try_loading, best_alpha, best_beta, 0)
    last_mean_loss = np.mean([sample[target_loss] for sample in samples])
    log_info("Initial loss: {}".format(last_mean_loss))

    # Search start
    alpha_radius = FLAGS.finetune_lm_alpha_radius
    beta_radius = FLAGS.finetune_lm_beta_radius
    n_iter = FLAGS.finetune_lm_n_iterations
    for _ in range(n_iter):
        coords = np.random.normal(size=2)
        length = np.math.sqrt(sum(coords**2))
        rand_points = coords / length

        alpha = best_alpha + rand_points[0] * alpha_radius
        beta = best_beta + rand_points[1] * beta_radius

        # constraint params in boundary
        alpha = max(min(alpha, alpha_max), alpha_min)
        beta = max(min(beta, beta_max), beta_min)

        tfv1.reset_default_graph()
        samples = evaluate(sample_csvs, create_model, try_loading, alpha, beta, 0)
        mean_loss = np.mean([sample[target_loss] for sample in samples])
        if mean_loss < last_mean_loss:
            log_info('Found lower loss: {} -> {}, set new position:'.format(last_mean_loss, mean_loss))
            log_info('Alpha = {} -> {}'.format(best_alpha, alpha))
            log_info('Beta = {} -> {}'.format(best_beta, beta))
            last_mean_loss = mean_loss
            best_alpha = alpha
            best_beta = beta

    # return the best alpha/beta and used parameters
    return {
        'best_alpha': best_alpha,
        'best_beta': best_beta,
        'target_mean_loss': target_loss,
        'alpha_start': alpha_start,
        'alpha_min': alpha_min,
        'alpha_max': alpha_max,
        'alpha_radius': alpha_radius,
        'beta_start': beta_start,
        'beta_min': beta_min,
        'beta_max': beta_max,
        'beta_radius': beta_radius,
        'n_iterations': n_iter,
    }


def finetune_lm_parabola(sample_csvs, create_model, try_loading):

    min_lm_alpha = FLAGS.finetune_lm_alpha_min
    max_lm_alpha = FLAGS.finetune_lm_alpha_max
    lm_alpha_steps = FLAGS.finetune_lm_alpha_steps

    min_lm_beta = FLAGS.finetune_lm_beta_min
    max_lm_beta = FLAGS.finetune_lm_beta_max
    lm_beta_steps = FLAGS.finetune_lm_beta_steps

    target_loss = FLAGS.finetune_lm_target_mean_loss

    # set beta at center first
    alpha = np.mean([min_lm_alpha, max_lm_alpha])

    def tune_best(iter_results, x_name, y_name, min_param, max_param):
        x_y_pairs = []
        for result in iter_results:
            x_y_pairs.append([result[x_name], result[y_name]])
        best_value, r_square = fit_parabola(x_y_pairs, min_param, max_param)
        return best_value, r_square

    # scan beta
    iter_results = iter_evaluate(sample_csvs, create_model, try_loading,
                                 [alpha],
                                 np.linspace(start=min_lm_beta, stop=max_lm_beta, num=lm_beta_steps))
    best_beta, r_square_beta = tune_best(iter_results, 'beta', target_loss, min_lm_beta, max_lm_beta)

    # scan alpha
    iter_results = iter_evaluate(sample_csvs, create_model, try_loading,
                                 np.linspace(start=min_lm_alpha, stop=max_lm_alpha, num=lm_alpha_steps),
                                 [best_beta])
    best_alpha, r_square_alpha = tune_best(iter_results, 'alpha', target_loss, min_lm_alpha, max_lm_alpha)

    # return the best alpha/beta and used parameters
    return {
        'best_alpha': best_alpha,
        'best_beta': best_beta,
        'r_square_alpha': r_square_alpha,
        'r_square_beta': r_square_beta,
        'target_mean_loss': target_loss,
        'alpha_min': min_lm_alpha,
        'alpha_max': max_lm_alpha,
        'alpha_steps': lm_alpha_steps,
        'beta_steps': lm_beta_steps,
    }


def finetune_parabola_and_random_search(sample_csvs, create_model, try_loading):
    result = finetune_lm_parabola(sample_csvs, create_model, try_loading)
    result_rand = finetune_random_search(sample_csvs, create_model, try_loading, result['best_alpha'], result['best_beta'])
    result.update(result_rand)
    return result


def iter_evaluate(test_csvs, create_model, try_loading, alphas, betas):
    all_samples = []
    for alpha in alphas:
        for beta in betas:

            alpha = max(alpha, 0.0)
            beta = max(beta, 0.0)

            tfv1.reset_default_graph()
            samples = evaluate(test_csvs, create_model, try_loading, alpha, beta, 0)

            # inject alpha beta
            for sample in samples:
                sample.update({
                    'alpha': alpha,
                    'beta': beta,
                })

            all_samples += samples
    return all_samples


def fit_parabola(x_y_pairs, param_min, param_max):
    # input: <list>
    # x_y_pairs = [
    # [x0, y0],
    # [x1, y1],
    # .
    # .
    # .
    # ]

    # assume:
    #   y0 = C0 * x0^2 + C1 * x0 + C2
    #   y1 = C0 * x1^2 + C1 * x1 + C2
    #   .
    #   .
    #   .
    # X = [[x^2, x, 1], [x^2, x1, 1], ...]
    # Y = [[y0], [y1], ...]
    # A = [[C0], [C1], [C2]]
    # XA = Y
    # if C0 > 0 => best_alpha = - C1 / 2 / C0
    # if C0 < 0 => best_alpha = X[argmin(result)]
    mat_x = []
    mat_y = []
    for x, y in x_y_pairs:
        mat_x.append([x**2, x, 1.0])
        mat_y.append([y])
    mat_a = np.matmul(np.linalg.pinv(mat_x), mat_y)
    c0 = mat_a[0, 0]
    c1 = mat_a[1, 0]

    mat_y_hat = np.matmul(mat_x, mat_a)
    r_square = np.mean((mat_y_hat - mat_y)**2)

    if c0 <= 0 or c1 >= 0:
        log_info("[Fitting Failed] it's not a valley parabola, so just pick a lowest parameter")
        return pick_lowest_param(x_y_pairs), None

    # the parabola has minimum param
    return max(min(- c1 / 2.0 / c0, param_max), param_min), r_square


def pick_lowest_param(x_y_pairs):
    df = pd.DataFrame([{'x': x, 'y': y} for x, y in x_y_pairs])
    return float(df.groupby('x').mean()['y'].idxmin())


VALID_METHODS = {
    'parabola': finetune_lm_parabola,
    'random_search': finetune_random_search,
    'parabola+random_search': finetune_parabola_and_random_search,
}
