#!/bin/bash

set -xe

arch=$1

source $(dirname "$0")/tc-tests-utils.sh

source $(dirname "$0")/tf_tc-vars.sh

BAZEL_TARGETS="
//native_client:libdeepspeech.so
"

if [ "${arch}" = "--arm64" ]; then
    BAZEL_BUILD_FLAGS="${BAZEL_IOS_ARM64_FLAGS}"
else
    BAZEL_BUILD_FLAGS="${BAZEL_IOS_X86_64_FLAGS}"
fi

BAZEL_ENV_FLAGS="TF_NEED_CUDA=0"

do_bazel_build

do_deepspeech_ios_framework_build "${arch}"
