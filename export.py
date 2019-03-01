#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from util.config import initialize_globals
from util.flags import create_flags, FLAGS
from util.logging import log_error


def main(_):
    initialize_globals()
    if not FLAGS.export_dir:
        log_error('You need to specify the export dir via the --export_dir flag.')
        exit(1)

    from DeepSpeech import export
    export()


if __name__ == '__main__':
    create_flags()
    tf.app.run(main)
