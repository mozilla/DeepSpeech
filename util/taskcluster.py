#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

if __name__ == '__main__':
    try:
        from mozilla_voice_stt_training.util import taskcluster as dsu_taskcluster
    except ImportError:
        print('Training package is not installed. See training documentation.')
        raise

    dsu_taskcluster.main()
