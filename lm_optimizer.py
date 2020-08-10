#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import absl.app
import optuna
import sys
import tensorflow.compat.v1 as tfv1

from mozilla_voice_stt_training.evaluate import evaluate
from mozilla_voice_stt_training.train import create_model
from mozilla_voice_stt_training.util.config import Config, initialize_globals
from mozilla_voice_stt_training.util.flags import create_flags, FLAGS
from mozilla_voice_stt_training.util.logging import log_error
from mozilla_voice_stt_training.util.evaluate_tools import wer_cer_batch
from ds_ctcdecoder import Scorer


def character_based():
    is_character_based = False
    if FLAGS.scorer_path:
        scorer = Scorer(FLAGS.lm_alpha, FLAGS.lm_beta, FLAGS.scorer_path, Config.alphabet)
        is_character_based = scorer.is_utf8_mode()
    return is_character_based

def objective(trial):
    FLAGS.lm_alpha = trial.suggest_uniform('lm_alpha', 0, FLAGS.lm_alpha_max)
    FLAGS.lm_beta = trial.suggest_uniform('lm_beta', 0, FLAGS.lm_beta_max)

    is_character_based = trial.study.user_attrs['is_character_based']

    samples = []
    for step, test_file in enumerate(FLAGS.test_files.split(',')):
        tfv1.reset_default_graph()

        current_samples = evaluate([test_file], create_model)
        samples += current_samples

        # Report intermediate objective value.
        wer, cer = wer_cer_batch(current_samples)
        trial.report(cer if is_character_based else wer, step)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    wer, cer = wer_cer_batch(samples)
    return cer if is_character_based else wer

def main(_):
    initialize_globals()

    if not FLAGS.test_files:
        log_error('You need to specify what files to use for evaluation via '
                  'the --test_files flag.')
        sys.exit(1)

    is_character_based = character_based()

    study = optuna.create_study()
    study.set_user_attr("is_character_based", is_character_based)
    study.optimize(objective, n_jobs=1, n_trials=FLAGS.n_trials)
    print('Best params: lm_alpha={} and lm_beta={} with WER={}'.format(study.best_params['lm_alpha'],
                                                                       study.best_params['lm_beta'],
                                                                       study.best_value))


if __name__ == '__main__':
    create_flags()
    absl.app.run(main)
