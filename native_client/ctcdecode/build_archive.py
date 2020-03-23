#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

import glob
import os
import shlex
import subprocess
import sys

from multiprocessing.dummy import Pool

if sys.platform.startswith('win'):
    ARGS = ['/nologo', '/D KENLM_MAX_ORDER=6', '/EHsc', '/source-charset:utf-8']
    OPT_ARGS = ['/O2', '/MT', '/D NDEBUG']
    DBG_ARGS = ['/Od', '/MTd', '/Zi', '/U NDEBUG', '/D DEBUG']
    OPENFST_DIR = 'third_party/openfst-1.6.9-win'
else:
    ARGS = ['-DKENLM_MAX_ORDER=6', '-std=c++11', '-Wno-unused-local-typedefs', '-Wno-sign-compare']
    OPT_ARGS = ['-O3', '-DNDEBUG']
    DBG_ARGS = ['-O0', '-g', '-UNDEBUG', '-DDEBUG']
    OPENFST_DIR = 'third_party/openfst-1.6.7'



INCLUDES = [
    '..',
    '../kenlm',
    OPENFST_DIR + '/src/include',
    'third_party/ThreadPool'
]

KENLM_FILES = (glob.glob('../kenlm/util/*.cc')
                + glob.glob('../kenlm/lm/*.cc')
                + glob.glob('../kenlm/util/double-conversion/*.cc'))

KENLM_FILES += glob.glob(OPENFST_DIR + '/src/lib/*.cc')

KENLM_FILES = [
    fn for fn in KENLM_FILES
    if not (fn.endswith('main.cc') or fn.endswith('test.cc') or fn.endswith(
        'unittest.cc'))
]

CTC_DECODER_FILES = [
    'ctc_beam_search_decoder.cpp',
    'scorer.cpp',
    'path_trie.cpp',
    'decoder_utils.cpp',
    'workspace_status.cc'
]

def build_archive(srcs=[], out_name='', build_dir='temp_build/temp_build', debug=False, num_parallel=1):
    compiler = os.environ.get('CXX', 'g++')
    ar = os.environ.get('AR', 'ar')
    libtool = os.environ.get('LIBTOOL', 'libtool')
    cflags = os.environ.get('CFLAGS', '') + os.environ.get('CXXFLAGS', '')
    args = ARGS + (DBG_ARGS if debug else OPT_ARGS)

    for file in srcs:
        outfile = os.path.join(build_dir, os.path.splitext(file)[0] + '.o')
        outdir = os.path.dirname(outfile)
        if not os.path.exists(outdir):
            print('mkdir', outdir)
            os.makedirs(outdir)

    def build_one(file):
        outfile = os.path.join(build_dir, os.path.splitext(file)[0] + '.o')
        if os.path.exists(outfile):
            return

        if sys.platform.startswith('win'):
            cmd = '"{cc}" -c {cflags} {args} {includes} {infile} -Fo"{outfile}"'.format(
                cc=compiler,
                cflags=cflags,
                args=' '.join(args),
                includes=' '.join('-I' + i for i in INCLUDES),
                infile=file,
                outfile=outfile,
            )
            cmd = cmd.replace('\\', '/')
        else:
            cmd = '{cc} -fPIC -c {cflags} {args} {includes} {infile} -o {outfile}'.format(
                cc=compiler,
                cflags=cflags,
                args=' '.join(args),
                includes=' '.join('-I' + i for i in INCLUDES),
                infile=file,
                outfile=outfile,
            )
        print(cmd)
        subprocess.check_call(shlex.split(cmd))
        return outfile

    pool = Pool(num_parallel)
    obj_files = list(pool.imap_unordered(build_one, srcs))

    if sys.platform.startswith('darwin'):
        cmd = '{libtool} -static -o {outfile} {infiles}'.format(
            libtool=libtool,
            outfile=out_name,
            infiles=' '.join(obj_files),
        )
        print(cmd)
        subprocess.check_call(shlex.split(cmd))
    elif sys.platform.startswith('win'):
        obj_files = [s for s in obj_files if s != None]
        cmd = '"lib.exe" /OUT:"{outfile}" {infiles} /MACHINE:X64 /NOLOGO'.format(
            ar=ar,
            outfile=out_name,
            infiles=' '.join(obj_files))
        cmd = cmd.replace('\\', '/')
        print(cmd)
        subprocess.check_call(shlex.split(cmd))
    else:
        cmd = '{ar} rcs {outfile} {infiles}'.format(
            ar=ar,
            outfile=out_name,
            infiles=' '.join(obj_files)
        )
        print(cmd)
        subprocess.check_call(shlex.split(cmd))

if __name__ == '__main__':
    build_common()
