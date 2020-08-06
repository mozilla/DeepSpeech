#!/bin/bash

set -xe

arm_flavor=$1

source $(dirname "$0")/tc-tests-utils.sh

source $(dirname "$0")/tf_tc-vars.sh

BAZEL_TARGETS="
//native_client:libmozilla_voice_stt.so
//native_client:generate_scorer_package
"

if [ "${arm_flavor}" = "armeabi-v7a" ]; then
    LOCAL_ANDROID_FLAGS="${BAZEL_ANDROID_ARM_FLAGS}"
fi

if [ "${arm_flavor}" = "arm64-v8a" ]; then
    LOCAL_ANDROID_FLAGS="${BAZEL_ANDROID_ARM64_FLAGS}"
fi

if [ "${arm_flavor}" = "x86_64" ]; then
    LOCAL_ANDROID_FLAGS="--config=android --cpu=x86_64 --action_env ANDROID_NDK_API_LEVEL=21 --cxxopt=-std=c++14 --copt=-D_GLIBCXX_USE_C99"
fi

BAZEL_BUILD_FLAGS="--define=runtime=tflite ${LOCAL_ANDROID_FLAGS} ${BAZEL_EXTRA_FLAGS}"
BAZEL_ENV_FLAGS="TF_NEED_CUDA=0"
SYSTEM_TARGET=
SYSTEM_RASPBIAN=

do_bazel_build

do_deepspeech_ndk_build "${arm_flavor}"
