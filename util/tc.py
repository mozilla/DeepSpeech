#!/usr/bin/env python
from __future__ import print_function
from __future__ import absolute_import
# -*- coding: utf-8 -*-

import sys
import os
import stat
import urllib

def maybe_download_tc(target_dir=None, tc_url=None, progress=True):
    def report_progress(count, block_size, total_size):
        percent = int((count * block_size * 100) / total_size)
        sys.stdout.write("\rDownloading: %d%%" % percent)
        sys.stdout.flush()

        if percent >= 100:
            print('\n')

    assert target_dir is not None

    target_dir = os.path.abspath(target_dir)
    assert os.path.isdir(os.path.dirname(target_dir))

    tc_filename = os.path.basename(tc_url)
    target_file = os.path.join(target_dir, tc_filename)
    if not os.path.isfile(target_file):
        print('Downloading %s ...' % tc_url)
        urllib.urlretrieve(tc_url, target_file, reporthook=(report_progress if progress else None))
    else:
        print('File already exists: %s' % target_file)

    return target_file

def maybe_download_tc_bin(**kwargs):
    final_file = maybe_download_tc(kwargs['target_dir'], kwargs['tc_url'], kwargs['progress'])
    final_stat = os.stat(final_file)
    os.chmod(final_file, final_stat.st_mode | stat.S_IEXEC)
