#!/bin/bash

set -xe

source $(dirname "$0")/tc-tests-utils.sh

source $(dirname "$0")/tf_tc-vars.sh

if [ "${OS}" = "${TC_MSYS_VERSION}" ]; then
  export SYSTEM_TARGET=host-win
else
  export SYSTEM_TARGET=host
fi;

do_deepspeech_decoder_build
