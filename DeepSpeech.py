#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import sys
import subprocess

def check_training_package():
    try:
        import deepspeech_training  # Check if the training package is installed
        return True
    except ImportError:
        return False

def run_training():
    try:
        from deepspeech_training import train as ds_train
        ds_train.run_script()  # Run the DeepSpeech training script
    except ImportError:
        print('Training package is not installed. See training documentation.')
    except Exception as e:
        print(f'An error occurred during training: {e}')

if __name__ == '__main__':
    if not check_training_package():
        print('DeepSpeech training package is not installed.')
        print('Please install it by following the training documentation.')
        sys.exit(1)

    run_training()
