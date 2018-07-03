#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import, division

import subprocess
import sys
import os
import errno
import stat
import six.moves.urllib as urllib

DEFAULT_SCHEMES = {
    'deepspeech': 'https://index.taskcluster.net/v1/task/project.deepspeech.deepspeech.native_client.%(branch_name)s.%(arch_string)s/artifacts/public/%(artifact_name)s',
    'tensorflow': 'https://index.taskcluster.net/v1/task/project.deepspeech.tensorflow.pip.%(branch_name)s.%(arch_string)s/artifacts/public/%(artifact_name)s'
}

TASKCLUSTER_SCHEME = os.getenv('TASKCLUSTER_SCHEME', DEFAULT_SCHEMES['deepspeech'])

def get_tc_url(arch_string=None, artifact_name='native_client.tar.xz', branch_name='master'):
    assert arch_string is not None
    assert artifact_name is not None
    assert len(artifact_name) > 0
    assert branch_name is not None
    assert len(branch_name) > 0

    return TASKCLUSTER_SCHEME % { 'arch_string': arch_string, 'artifact_name': artifact_name, 'branch_name': branch_name}

def maybe_download_tc(target_dir, tc_url, progress=True):
    def report_progress(count, block_size, total_size):
        percent = (count * block_size * 100) // total_size
        sys.stdout.write("\rDownloading: %d%%" % percent)
        sys.stdout.flush()

        if percent >= 100:
            print('\n')

    assert target_dir is not None

    target_dir = os.path.abspath(target_dir)
    try:
        os.makedirs(target_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise e
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
    import argparse

    parser = argparse.ArgumentParser(description='Tooling to ease downloading of components from TaskCluster.')
    parser.add_argument('--target', required=True,
                        help='Where to put the native client binary files')
    parser.add_argument('--arch', required=False, default='cpu',
                        help='Which architecture to download binaries for. "arm" for ARM 7 (32-bit), "gpu" for CUDA enabled x86_64 binaries, "cpu" for CPU-only x86_64 binaries, "osx" for CPU-only x86_64 OSX binaries. Optional ("cpu" by default)')
    parser.add_argument('--artifact', required=False,
                        default='native_client.tar.xz',
                        help='Name of the artifact to download. Defaults to "native_client.tar.xz"')
    parser.add_argument('--source', required=False, default=None,
                        help='Name of the TaskCluster scheme to use.')
    parser.add_argument('--branch', required=False, default='master',
                        help='Branch name to use. Defaulting to "master".')

    args = parser.parse_args()

    if args.source is not None:
        if args.source in DEFAULT_SCHEMES:
            TASKCLUSTER_SCHEME = DEFAULT_SCHEMES[args.source]
        else:
            print('No such scheme: %s' % args.source)
            exit(1)

    maybe_download_tc(target_dir=args.target, tc_url=get_tc_url(args.arch, args.artifact, args.branch))

    if '.tar.' in args.artifact:
        subprocess.check_call(['tar', 'xvf', os.path.join(args.target, args.artifact), '-C', args.target])
