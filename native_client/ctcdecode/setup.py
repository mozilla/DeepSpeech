#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

from distutils.command.build import build
from setuptools import setup, Extension, distutils

import argparse
import multiprocessing.pool
import os
import platform
import sys

if sys.platform.startswith('win'):
    ARGS = ['/nologo', '/D KENLM_MAX_ORDER=6', '/EHsc', '/source-charset:utf-8']
    OPT_ARGS = ['/O2', '/MT', '/D NDEBUG']
    DBG_ARGS = ['/Od', '/MTd', '/Zi', '/U NDEBUG', '/D DEBUG']
    OPENFST_DIR = 'third_party/openfst-1.6.9-win'
else:
    ARGS = ['-std=c++11']
    OPT_ARGS = ['-O3', '-DNDEBUG']
    DBG_ARGS = ['-O0', '-g', '-UNDEBUG', '-DDEBUG']
    OPENFST_DIR = 'third_party/openfst-1.6.7'

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

debug = '--debug' in sys.argv

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


project_version = read('../../training/deepspeech_training/VERSION').strip()

decoder_module = Extension(
    name='ds_ctcdecoder._swigwrapper',
    sources=['swigwrapper.i'],
    library_dirs=[os.path.join(os.environ['TFDIR'], 'bazel-bin', 'native_client')],
    libraries=['decoder', 'kenlm', 'ds_version'],
    swig_opts=['-c++', '-extranative'],
    language='c++',
    include_dirs=[numpy_include, '..', OPENFST_DIR + '/src/include'],
    extra_compile_args=ARGS + (DBG_ARGS if debug else OPT_ARGS),
)

class BuildExtFirst(build):
    sub_commands = [('build_ext', build.has_ext_modules),
                    ('build_py', build.has_pure_modules),
                    ('build_clib', build.has_c_libraries),
                    ('build_scripts', build.has_scripts)]

setup(
    name='ds_ctcdecoder',
    version=project_version,
    description="""DS CTC decoder""",
    cmdclass = {'build': BuildExtFirst},
    ext_modules=[decoder_module],
    package_dir = {'ds_ctcdecoder': '.'},
    py_modules=['ds_ctcdecoder', 'ds_ctcdecoder.swigwrapper'],
    install_requires = ['numpy%s' % numpy_min_ver],
)
