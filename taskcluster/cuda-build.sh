#!/bin/bash

set -xe

source $(dirname "$0")/tc-tests-utils.sh

source $(dirname "$0")/tf_tc-vars.sh

BAZEL_TARGETS="
//native_client:libmozilla_voice_stt.so
//native_client:generate_scorer_package
"

BAZEL_ENV_FLAGS="TF_NEED_CUDA=1 ${TF_CUDA_FLAGS}"
BAZEL_BUILD_FLAGS="${BAZEL_CUDA_FLAGS} ${BAZEL_EXTRA_FLAGS} ${BAZEL_OPT_FLAGS}"
SYSTEM_TARGET=host
EXTRA_LOCAL_CFLAGS=""
EXTRA_LOCAL_LDFLAGS="-L${DS_ROOT_TASK}/DeepSpeech/CUDA/lib64/ -L${DS_ROOT_TASK}/DeepSpeech/CUDA/lib64/stubs/ -lcudart -lcuda"

do_bazel_build

do_deepspeech_binary_build

do_deepspeech_python_build "--cuda"

do_deepspeech_nodejs_build "--cuda"
