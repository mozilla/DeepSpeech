#!/bin/bash

set -xe

source $(dirname "$0")/../tc-tests-utils.sh

source ${DS_ROOT_TASK}/DeepSpeech/tf/tc-vars.sh

BAZEL_ENV_FLAGS="TF_NEED_CUDA=0"
BAZEL_TARGETS="${BAZEL_AOT_TARGETS}"

do_get_model_parameters "${DEEPSPEECH_TEST_MODEL}" AOT_MODEL_PARAMS
BAZEL_BUILD_FLAGS="${BAZEL_OPT_FLAGS} ${BAZEL_AOT_BUILD_FLAGS} ${AOT_MODEL_PARAMS}"

do_bazel_build
