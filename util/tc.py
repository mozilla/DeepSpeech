#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import, division

import subprocess
import sys
import os
import stat
import six.moves.urllib as urllib

TASKCLUSTER_SCHEME = os.getenv('TASKCLUSTER_SCHEME',
                               'https://index.taskcluster.net/v1/task/project.deepspeech.deepspeech.native_client.master.%(arch_string)s/artifacts/public/native_client.tar.xz')

def get_tc_url(arch_string=None):
    assert arch_string is not None

    return TASKCLUSTER_SCHEME % { 'arch_string': arch_string }

def maybe_download_tc(target_dir, tc_url, progress=True):
    def report_progress(count, block_size, total_size):
        percent = (count * block_size * 100) // total_size
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
        urllib.request.urlretrieve(tc_url, target_file, reporthook=(report_progress if progress else None))
    else:
        print('File already exists: %s' % target_file)

    return target_file

def maybe_download_tc_bin(**kwargs):
    final_file = maybe_download_tc(kwargs['target_dir'], kwargs['tc_url'], kwargs['progress'])
    final_stat = os.stat(final_file)
    os.chmod(final_file, final_stat.st_mode | stat.S_IEXEC)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Downloads and extracts native client binaries.')
        print('Usage: {} <target_dir> [arch_string]'.format(sys.argv[0]))
        print('  <target_dir>: Where to put the native client binary files')
        print('  [arch_string]: Which architecture to download binaries for. "arm" for ARM 7 (32-bit), "gpu" for CUDA enabled x86_64 binaries, "cpu" for CPU-only x86_64 binaries. Optional ("cpu" by default)')
        exit(1)

    if len(sys.argv) == 2:
        arch_string = 'cpu'
    else:
        arch_string = sys.argv[2]

    target_dir = sys.argv[1]

    maybe_download_tc(target_dir=target_dir, tc_url=get_tc_url(arch_string))

    subprocess.check_call(['tar', 'xvzf', os.path.join(target_dir, 'native_client.tar.xz'), '-C', target_dir])
