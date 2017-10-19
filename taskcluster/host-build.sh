#!/bin/bash

set -xe

source $(dirname "$0")/../tc-tests-utils.sh

source ${DS_ROOT_TASK}/DeepSpeech/tf/tc-vars.sh

BAZEL_TARGETS="
${BAZEL_AOT_TARGETS}
//native_client:deepspeech
//native_client:deepspeech_utils
//native_client:generate_trie
${BAZEL_CTC_TARGETS}
"

BAZEL_ENV_FLAGS="TF_NEED_CUDA=0"
SYSTEM_TARGET=host

EXTRA_LOCAL_CFLAGS="${EXTRA_AOT_CFLAGS}"
EXTRA_LOCAL_LDFLAGS="${EXTRA_AOT_LDFLAGS}"
EXTRA_LOCAL_LIBS="${EXTRA_AOT_LIBS}"

do_get_model_parameters "${DEEPSPEECH_PROD_MODEL}" AOT_MODEL_PARAMS
BAZEL_BUILD_FLAGS="${BAZEL_OPT_FLAGS} ${BAZEL_AOT_BUILD_FLAGS} ${AOT_MODEL_PARAMS}"

do_bazel_build

do_deepspeech_binary_build

do_deepspeech_python_build

do_deepspeech_nodejs_build
