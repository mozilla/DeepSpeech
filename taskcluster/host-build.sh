#!/bin/bash

set -xe

source $(dirname "$0")/../tc-tests-utils.sh

source ${DS_ROOT_TASK}/DeepSpeech/tf/tc-vars.sh

BAZEL_TARGETS="
//native_client:libdeepspeech.so
//native_client:generate_trie
"

BAZEL_BUILD_FLAGS="${BAZEL_OPT_FLAGS} ${BAZEL_EXTRA_FLAGS}"
BAZEL_ENV_FLAGS="TF_NEED_CUDA=0"
SYSTEM_TARGET=host

do_bazel_build

do_deepspeech_binary_build

do_deepspeech_python_build

do_deepspeech_nodejs_build

