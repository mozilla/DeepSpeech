#!/bin/bash

set -xe

runtime=$1

source $(dirname "$0")/tc-tests-utils.sh

source $(dirname "$0")/tf_tc-vars.sh

BAZEL_TARGETS="
//native_client:libdeepspeech.so
//native_client:generate_scorer_package
"

if [ "${runtime}" = "tflite" ]; then
  BAZEL_BUILD_TFLITE="--define=runtime=tflite"
fi;
BAZEL_BUILD_FLAGS="${BAZEL_BUILD_TFLITE} ${BAZEL_OPT_FLAGS} ${BAZEL_EXTRA_FLAGS}"

BAZEL_ENV_FLAGS="TF_NEED_CUDA=0"
SYSTEM_TARGET=host

do_bazel_build

do_deepspeech_binary_build

if [ "${runtime}" = "tflite" ]; then
  do_deepspeech_python_build "--tflite"
else
  do_deepspeech_python_build
fi

do_deepspeech_nodejs_build

