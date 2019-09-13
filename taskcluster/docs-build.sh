#!/bin/bash

set -xe

THIS=$(dirname "$0")

pushd ${THIS}/../
  export PATH=$HOME/.local/bin:${THIS}/../doc/node_modules/.bin/:$PATH
  make -C doc/ html dist
popd
