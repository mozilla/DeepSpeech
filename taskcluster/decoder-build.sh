#!/bin/bash

set -xe

source $(dirname "$0")/../tc-tests-utils.sh

source ${DS_ROOT_TASK}/DeepSpeech/tf/tc-vars.sh

BAZEL_ENV_FLAGS="TF_NEED_CUDA=0"
BAZEL_BUILD_FLAGS="${BAZEL_OPT_FLAGS} ${BAZEL_EXTRA_FLAGS}"
BAZEL_TARGETS="${BAZEL_CTC_TARGETS}"

do_bazel_shared_build
