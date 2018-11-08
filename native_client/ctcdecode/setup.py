#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

from distutils.command.build import build
from setuptools import setup, Extension, distutils

import argparse
import glob
import multiprocessing.pool
import os
import platform
import sys

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
args = parser.parse_known_args()

# reconstruct sys.argv to pass to setup below
sys.argv = [sys.argv[0]] + args[1]

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

project_version = read('../../VERSION').strip()

# monkey-patch for parallel compilation
# See: https://stackoverflow.com/a/13176803
def parallelCCompile(self,
                     sources,
                     output_dir=None,
                     macros=None,
                     include_dirs=None,
                     debug=0,
                     extra_preargs=None,
                     extra_postargs=None,
                     depends=None):
    # those lines are copied from distutils.ccompiler.CCompiler directly
    macros, objects, extra_postargs, pp_opts, build = self._setup_compile(
        output_dir, macros, include_dirs, sources, depends, extra_postargs)
    cc_args = self._get_cc_args(pp_opts, debug, extra_preargs)

    # parallel code
    def _single_compile(obj):
        try:
            src, ext = build[obj]
        except KeyError:
            return
        self._compile(obj, src, ext, cc_args, extra_postargs, pp_opts)

    # convert to list, imap is evaluated on-demand
    thread_pool = multiprocessing.pool.ThreadPool(args[0].num_processes)
    list(thread_pool.imap(_single_compile, objects))
    return objects

# hack compile to support parallel compiling
distutils.ccompiler.CCompiler.compile = parallelCCompile

FILES = glob.glob('../kenlm/util/*.cc') \
        + glob.glob('../kenlm/lm/*.cc') \
        + glob.glob('../kenlm/util/double-conversion/*.cc')

FILES += glob.glob('third_party/openfst-1.6.7/src/lib/*.cc')

FILES = [
    fn for fn in FILES
    if not (fn.endswith('main.cc') or fn.endswith('test.cc') or fn.endswith(
        'unittest.cc'))
]

ARGS = ['-O3', '-DNDEBUG', '-DKENLM_MAX_ORDER=6', '-std=c++11',
        '-Wno-unused-local-typedef', '-Wno-sign-compare']

decoder_module = Extension(
    name='ds_ctcdecoder._swigwrapper',
    sources=['swigwrapper.i'] + FILES + glob.glob('*.cpp'),
    swig_opts=['-c++', '-extranative'],
    language='c++',
    include_dirs=[
        numpy_include,
        '..',
        '../kenlm',
        'third_party/openfst-1.6.7/src/include',
        'third_party/ThreadPool',
    ],
    extra_compile_args=ARGS
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
