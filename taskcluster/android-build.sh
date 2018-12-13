#!/bin/bash

set -xe

arm_flavor=$1

source $(dirname "$0")/../tc-tests-utils.sh

source ${DS_ROOT_TASK}/DeepSpeech/tf/tc-vars.sh

BAZEL_TARGETS="
//native_client:libdeepspeech.so
"

if [ "${arm_flavor}" = "armeabi-v7a" ]; then
    LOCAL_ANDROID_FLAGS="${BAZEL_ANDROID_ARM_FLAGS}"
fi

if [ "${arm_flavor}" = "arm64-v8a" ]; then
    LOCAL_ANDROID_FLAGS="${BAZEL_ANDROID_ARM64_FLAGS}"
fi

BAZEL_BUILD_FLAGS="${LOCAL_ANDROID_FLAGS} ${BAZEL_EXTRA_FLAGS}"
BAZEL_ENV_FLAGS="TF_NEED_CUDA=0"
SYSTEM_TARGET=
SYSTEM_RASPBIAN=

do_bazel_build

do_deepspeech_ndk_build "${arm_flavor}"
