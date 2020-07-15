#!/bin/bash

set -xe

source $(dirname "$0")/tc-tests-utils.sh

source $(dirname "$0")/tf_tc-vars.sh

BAZEL_TARGETS="
//native_client:libdeepspeech.so
"

BAZEL_BUILD_FLAGS="--config=ios_arm64 --define=runtime=tflite ${BAZEL_EXTRA_FLAGS}"

BAZEL_ENV_FLAGS="TF_NEED_CUDA=0"

do_bazel_build
