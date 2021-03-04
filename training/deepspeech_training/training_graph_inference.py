#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys

LOG_LEVEL_INDEX = sys.argv.index('--log_level') + 1 if '--log_level' in sys.argv else 0
DESIRED_LOG_LEVEL = sys.argv[LOG_LEVEL_INDEX] if 0 < LOG_LEVEL_INDEX < len(sys.argv) else '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = DESIRED_LOG_LEVEL

import absl.app
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tfv1

from ds_ctcdecoder import ctc_beam_search_decoder, Scorer
from .deepspeech_model import create_inference_graph, create_overlapping_windows
from .util.checkpoints import load_graph_for_evaluation
from .util.config import Config, initialize_globals
from .util.feeding import audiofile_to_features
from .util.flags import create_flags, FLAGS
from .util.logging import log_error


def do_single_file_inference(input_file_path):
    with tfv1.Session(config=Config.session_config) as session:
        inputs, outputs, _ = create_inference_graph(batch_size=1, n_steps=-1)

        # Restore variables from training checkpoint
        load_graph_for_evaluation(session)

        features, features_len = audiofile_to_features(input_file_path)
        previous_state_c = np.zeros([1, Config.n_cell_dim])
        previous_state_h = np.zeros([1, Config.n_cell_dim])

        # Add batch dimension
        features = tf.expand_dims(features, 0)
        features_len = tf.expand_dims(features_len, 0)

        # Evaluate
        features = create_overlapping_windows(features).eval(session=session)
        features_len = features_len.eval(session=session)

        probs = outputs['outputs'].eval(feed_dict={
            inputs['input']: features,
            inputs['input_lengths']: features_len,
            inputs['previous_state_c']: previous_state_c,
            inputs['previous_state_h']: previous_state_h,
        }, session=session)

        probs = np.squeeze(probs)

        if FLAGS.scorer_path:
            scorer = Scorer(FLAGS.lm_alpha, FLAGS.lm_beta,
                            FLAGS.scorer_path, Config.alphabet)
        else:
            scorer = None
        decoded = ctc_beam_search_decoder(probs, Config.alphabet, FLAGS.beam_width,
                                          scorer=scorer, cutoff_prob=FLAGS.cutoff_prob,
                                          cutoff_top_n=FLAGS.cutoff_top_n)
        # Print highest probability result
        print(decoded[0][1])


def main(_):
    initialize_globals()

    if FLAGS.one_shot_infer:
        tfv1.reset_default_graph()
        do_single_file_inference(FLAGS.one_shot_infer)
    else:
        log_error('Calling training_graph_inference script directly but no --one_shot_infer input audio file specified')
        sys.exit(1)


if __name__ == '__main__':
    create_flags()
    absl.app.run(main)
