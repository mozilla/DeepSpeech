#! /usr/bin/env python

from setuptools import setup, Extension
from distutils.command.build import build

import os
import numpy
import subprocess

try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

class BuildExtFirst(build):
    sub_commands = [('build_ext', build.has_ext_modules),
                    ('build_py', build.has_pure_modules),
                    ('build_clib', build.has_c_libraries),
                    ('build_scripts', build.has_scripts)]

deepspeech = Extension('_deepspeech',
        ['python/deepspeech.i'],
        include_dirs = [numpy_include],
        libraries = ['tensorflow', 'deepspeech', 'c_speech_features', 'kissfft'])

setup(name = 'deepspeech',
      description = 'A library for running inference on a DeepSpeech model',
      author = 'Chris Lord',
      author_email='chrislord.net@gmail.com',
      version = '0.0.1',
      package_dir = {'deepspeech': 'python'},
      packages = [ 'deepspeech' ],
      cmdclass = { 'build': BuildExtFirst },
      license = 'MPL-2.0',
      url = 'https://github.com/mozilla/DeepSpeech',
      ext_modules = [deepspeech])
