#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

from distutils.command.build import build
from setuptools import setup, Extension, distutils

import argparse
import multiprocessing.pool
import os
import platform
import sys

from build_common import *

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

build_dir = 'temp_build/temp_build'
common_build = 'common.a'

if not os.path.exists(common_build):
    if not os.path.exists(build_dir):
        os.makedirs(build_dir)

    build_common(out_name='common.a',
                 build_dir=build_dir,
                 num_parallel=args[0].num_processes)

decoder_module = Extension(
    name='ds_ctcdecoder._swigwrapper',
    sources=['swigwrapper.i'],
    swig_opts=['-c++', '-extranative'],
    language='c++',
    include_dirs=INCLUDES + [numpy_include],
    extra_compile_args=ARGS,
    extra_link_args=[common_build],
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
