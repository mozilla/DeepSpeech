#!/bin/bash

set -ex

# Taken from https://www.tensorflow.org/install/source
# Only future is needed for our builds, as we don't build the Python package
pip install -U --user future==0.17.1
