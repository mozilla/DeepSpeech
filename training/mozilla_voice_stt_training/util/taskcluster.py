#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import, division

import argparse
import errno
import gzip
import os
import platform
import six.moves.urllib as urllib
import stat
import subprocess
import sys

from pkg_resources import parse_version


DEFAULT_SCHEMES = {
    'deepspeech': 'https://community-tc.services.mozilla.com/api/index/v1/task/project.mozilla-voice-stt.native_client.%(branch_name)s.%(arch_string)s/artifacts/public/%(artifact_name)s',
    'tensorflow': 'https://community-tc.services.mozilla.com/api/index/v1/task/project.mozilla-voice-stt.tensorflow.pip.%(branch_name)s.%(arch_string)s/artifacts/public/%(artifact_name)s',
    'tensorflow_r1.15': 'https://community-tc.services.mozilla.com/api/index/v1/task/project.deepspeech.tensorflow.pip.%(branch_name)s.%(arch_string)s/artifacts/public/%(artifact_name)s'
}

TASKCLUSTER_SCHEME = os.getenv('TASKCLUSTER_SCHEME', DEFAULT_SCHEMES['deepspeech'])

def get_tc_url(arch_string, artifact_name='native_client.tar.xz', branch_name='master'):
    assert arch_string is not None
    assert artifact_name is not None
    assert artifact_name
    assert branch_name is not None
    assert branch_name

    return TASKCLUSTER_SCHEME % {'arch_string': arch_string, 'artifact_name': artifact_name, 'branch_name': branch_name}

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
    is_gzip = False
    if not os.path.isfile(target_file):
        print('Downloading %s ...' % tc_url)
        _, headers = urllib.request.urlretrieve(tc_url, target_file, reporthook=(report_progress if progress else None))
        is_gzip = headers.get('Content-Encoding') == 'gzip'
    else:
        print('File already exists: %s' % target_file)

    if is_gzip:
        with open(target_file, "r+b") as frw:
            decompressed = gzip.decompress(frw.read())
            frw.seek(0)
            frw.write(decompressed)
            frw.truncate()

    return target_file

def maybe_download_tc_bin(**kwargs):
    final_file = maybe_download_tc(kwargs['target_dir'], kwargs['tc_url'], kwargs['progress'])
    final_stat = os.stat(final_file)
    os.chmod(final_file, final_stat.st_mode | stat.S_IEXEC)

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

def main():
    parser = argparse.ArgumentParser(description='Tooling to ease downloading of components from TaskCluster.')
    parser.add_argument('--target', required=False,
                        help='Where to put the native client binary files')
    parser.add_argument('--arch', required=False,
                        help='Which architecture to download binaries for. "arm" for ARM 7 (32-bit), "arm64" for ARM64, "gpu" for CUDA enabled x86_64 binaries, "cpu" for CPU-only x86_64 binaries, "osx" for CPU-only x86_64 OSX binaries. Optional ("cpu" by default)')
    parser.add_argument('--artifact', required=False,
                        default='native_client.tar.xz',
                        help='Name of the artifact to download. Defaults to "native_client.tar.xz"')
    parser.add_argument('--source', required=False, default=None,
                        help='Name of the TaskCluster scheme to use.')
    parser.add_argument('--branch', required=False,
                        help='Branch name to use. Defaulting to current content of VERSION file.')

    args = parser.parse_args()

    if not args.target and not args.decoder:
        print('Pass either --target or --decoder.')
        sys.exit(1)

    is_arm = 'arm' in platform.machine()
    is_mac = 'darwin' in sys.platform
    is_64bit = sys.maxsize > (2**31 - 1)
    is_ucs2 = sys.maxunicode < 0x10ffff

    if not args.arch:
        if is_arm:
            args.arch = 'arm64' if is_64bit else 'arm'
        elif is_mac:
            args.arch = 'osx'
        else:
            args.arch = 'cpu'

    if not args.branch:
        version_string = read('../VERSION').strip()
        ds_version = parse_version(version_string)
        args.branch = "v{}".format(version_string)
    else:
        ds_version = parse_version(args.branch)

    if args.source is not None:
        if args.source in DEFAULT_SCHEMES:
            global TASKCLUSTER_SCHEME
            TASKCLUSTER_SCHEME = DEFAULT_SCHEMES[args.source]
        else:
            print('No such scheme: %s' % args.source)
            sys.exit(1)

    if args.artifact == "convert_graphdef_memmapped_format" and args.branch == "r1.15":
        TASKCLUSTER_SCHEME = DEFAULT_SCHEMES["tensorflow_r1.15"]

    maybe_download_tc(target_dir=args.target, tc_url=get_tc_url(args.arch, args.artifact, args.branch))

    if args.artifact == "convert_graphdef_memmapped_format":
        convert_graph_file = os.path.join(args.target, args.artifact)
        final_stat = os.stat(convert_graph_file)
        os.chmod(convert_graph_file, final_stat.st_mode | stat.S_IEXEC)

    if '.tar.' in args.artifact:
        subprocess.check_call(['tar', 'xvf', os.path.join(args.target, args.artifact), '-C', args.target])

if __name__ == '__main__':
    main()
