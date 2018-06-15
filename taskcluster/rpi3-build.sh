#!/bin/bash

set -xe

source $(dirname "$0")/../tc-tests-utils.sh

source ${DS_ROOT_TASK}/DeepSpeech/tf/tc-vars.sh

BAZEL_TARGETS="
//native_client:libdeepspeech.so
//native_client:deepspeech_utils
//native_client:generate_trie
"

BAZEL_BUILD_FLAGS="${BAZEL_ARM_FLAGS} ${BAZEL_EXTRA_FLAGS}"
BAZEL_ENV_FLAGS="TF_NEED_CUDA=0"
SYSTEM_TARGET=rpi3
SYSTEM_RASPBIAN=/tmp/multistrap-raspbian-stretch

if [ $1 = "--aot" ]; then
  EXTRA_LOCAL_CFLAGS="${EXTRA_AOT_CFLAGS}"
  EXTRA_LOCAL_LDFLAGS="${EXTRA_AOT_LDFLAGS}"
  EXTRA_LOCAL_LIBS="${EXTRA_AOT_LIBS}"

  do_get_model_parameters "${DEEPSPEECH_PROD_MODEL}" AOT_MODEL_PARAMS

  BAZEL_TARGETS="${BAZEL_AOT_TARGETS} ${BAZEL_TARGETS}"
  BAZEL_BUILD_FLAGS="${BAZEL_BUILD_FLAGS} ${BAZEL_AOT_BUILD_FLAGS} ${AOT_MODEL_PARAMS}"
fi;

maybe_install_xldd

do_bazel_build

do_deepspeech_binary_build

export SUPPORTED_PYTHON_VERSIONS="3.4.8:ucs4 3.5.3:ucs4"
do_deepspeech_python_build

do_deepspeech_nodejs_build
