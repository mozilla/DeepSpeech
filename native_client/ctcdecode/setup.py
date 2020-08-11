#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

from distutils.command.build import build
from setuptools import setup, Extension, distutils

import argparse
import multiprocessing.pool
import os
import platform
import sys

from build_archive import *

try:
    import numpy
    try:
        numpy_include = numpy.get_include()
    except AttributeError:
        numpy_include = numpy.get_numpy_include()
except ImportError:
    numpy_include = ''
    assert 'NUMPY_INCLUDE' in os.environ

numpy_include = os.getenv('NUMPY_INCLUDE', numpy_include)
numpy_min_ver = os.getenv('NUMPY_DEP_VERSION', '')

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--num_processes",
    default=1,
    type=int,
    help="Number of cpu processes to build package. (default: %(default)d)")
known_args, unknown_args = parser.parse_known_args()
debug = '--debug' in unknown_args

# reconstruct sys.argv to pass to setup below
sys.argv = [sys.argv[0]] + unknown_args

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

def maybe_rebuild(srcs, out_name, build_dir):
    if not os.path.exists(out_name):
        if not os.path.exists(build_dir):
            os.makedirs(build_dir)

        build_archive(srcs=srcs,
                     out_name=out_name,
                     build_dir=build_dir,
                     num_parallel=known_args.num_processes,
                     debug=debug)

project_version = read('../../training/mozilla_voice_stt_training/VERSION').strip()

build_dir = 'temp_build/temp_build'

if sys.platform.startswith('win'):
    archive_ext = 'lib'
else:
    archive_ext = 'a'

third_party_build = 'third_party.{}'.format(archive_ext)
ctc_decoder_build = 'first_party.{}'.format(archive_ext)


maybe_rebuild(KENLM_FILES, third_party_build, build_dir)
maybe_rebuild(CTC_DECODER_FILES, ctc_decoder_build, build_dir)

decoder_module = Extension(
    name='mvs_ctcdecoder._swigwrapper',
    sources=['swigwrapper.i'],
    swig_opts=['-c++', '-extranative'],
    language='c++',
    include_dirs=INCLUDES + [numpy_include],
    extra_compile_args=ARGS + (DBG_ARGS if debug else OPT_ARGS),
    extra_link_args=[ctc_decoder_build, third_party_build],
)

class BuildExtFirst(build):
    sub_commands = [('build_ext', build.has_ext_modules),
                    ('build_py', build.has_pure_modules),
                    ('build_clib', build.has_c_libraries),
                    ('build_scripts', build.has_scripts)]

setup(
    name='mvs_ctcdecoder',
    version=project_version,
    description="""DS CTC decoder""",
    cmdclass = {'build': BuildExtFirst},
    ext_modules=[decoder_module],
    package_dir = {'mvs_ctcdecoder': '.'},
    py_modules=['mvs_ctcdecoder', 'mvs_ctcdecoder.swigwrapper'],
    install_requires = ['numpy%s' % numpy_min_ver],
)
