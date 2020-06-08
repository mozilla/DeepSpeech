#!/bin/bash

set -xe

THIS=$(dirname "$0")

pushd ${THIS}/../
  make -C doc/ dist
popd
