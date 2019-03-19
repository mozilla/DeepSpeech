#!/bin/bash

set -xe

cuda=$1

source $(dirname "$0")/../tc-tests-utils.sh

source ${DS_ROOT_TASK}/DeepSpeech/tf/tc-vars.sh

BAZEL_TARGETS="
//native_client:libdeepspeech.so
//native_client:generate_trie
"

if [ "${cuda}" = "--cuda" ]; then
    BAZEL_ENV_FLAGS="TF_NEED_CUDA=1 ${TF_CUDA_FLAGS}"
    BAZEL_BUILD_FLAGS="${BAZEL_CUDA_FLAGS} ${BAZEL_EXTRA_FLAGS} ${BAZEL_OPT_FLAGS}"
    PROJECT_NAME="DeepSpeech-GPU"
else
    PROJECT_NAME="DeepSpeech"
    BAZEL_BUILD_FLAGS="${BAZEL_OPT_FLAGS} ${BAZEL_EXTRA_FLAGS}"
    BAZEL_ENV_FLAGS="TF_NEED_CUDA=0"
fi

SYSTEM_TARGET=host-win

do_bazel_build

if [ "${cuda}" = "--cuda" ]; then
    cp ${DS_ROOT_TASK}/DeepSpeech/tf/bazel-bin/native_client/liblibdeepspeech.so.ifso ${DS_ROOT_TASK}/DeepSpeech/tf/bazel-bin/native_client/libdeepspeech.so.if.lib
fi

export PATH=$PATH:$(cygpath ${ChocolateyInstall})/bin

do_deepspeech_binary_build

do_deepspeech_netframework_build

do_nuget_build "${PROJECT_NAME}"

shutdown_bazel
