#!/bin/bash

set -xe

source $(dirname "$0")/tc-tests-utils.sh

source $(dirname "$0")/tf_tc-vars.sh

BAZEL_TARGETS="
//native_client:libmozilla_voice_stt.so
//native_client:generate_scorer_package
"

BAZEL_BUILD_FLAGS="${BAZEL_ARM64_FLAGS} ${BAZEL_EXTRA_FLAGS}"
BAZEL_ENV_FLAGS="TF_NEED_CUDA=0"
SYSTEM_TARGET=rpi3-armv8
SYSTEM_RASPBIAN=/tmp/multistrap-armbian64-buster

maybe_install_xldd

do_bazel_build

do_deepspeech_binary_build

export SUPPORTED_PYTHON_VERSIONS="3.7.6:ucs2"
do_deepspeech_python_build

do_deepspeech_nodejs_build
