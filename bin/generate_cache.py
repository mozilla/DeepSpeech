#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

# Make sure we can import stuff from util/
# This script needs to be run from the root of the DeepSpeech repository
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import absl
import progressbar
import tensorflow as tf
import tensorflow.compat.v1 as tfv1
tfv1.enable_eager_execution()

from util.config import initialize_globals
from util.flags import create_flags, FLAGS
from util.logging import create_progressbar
from util.feeding import create_dataset

def main(_):
    initialize_globals()

    if not FLAGS.train_files:
        log_error('You need to specify training files to preprocess/cache via '
                  'the --train_files flag.')
        sys.exit(1)

    if not FLAGS.feature_cache:
        log_error('You need to specify where to create the feature cache via '
                  'the --feature_cache flag.')
        sys.exit(1)

    train_set = create_dataset(FLAGS.train_files.split(','),
                               batch_size=1,
                               enable_cache=True,
                               cache_path=FLAGS.feature_cache)

    bar = create_progressbar(prefix='Preprocessing | ',
                             widgets=['Steps: ', progressbar.Counter(), ' | ', progressbar.Timer()]).start()

    for i, _ in enumerate(train_set, start=1):
        bar.update(i)

    bar.finish()

if __name__ == '__main__':
    create_flags()
    absl.app.run(main)
