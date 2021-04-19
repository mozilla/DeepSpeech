#!/bin/bash

set -xe

source $(dirname "$0")/all-vars.sh
source $(dirname "$0")/all-utils.sh
source $(dirname "$0")/build-utils.sh

source $(dirname "$0")/tf-vars.sh

BAZEL_TARGETS="
//native_client:libdeepspeech.so
//native_client:generate_scorer_package
"

BAZEL_BUILD_FLAGS="${BAZEL_ARM_FLAGS} ${BAZEL_EXTRA_FLAGS}"
BAZEL_ENV_FLAGS="TF_NEED_CUDA=0"

maybe_install_xldd

do_bazel_build

do_deepspeech_binary_build
