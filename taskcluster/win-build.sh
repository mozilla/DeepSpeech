#!/bin/bash

set -xe

package_option=$1

source $(dirname "$0")/tc-tests-utils.sh

source $(dirname "$0")/tf_tc-vars.sh

BAZEL_TARGETS="
//native_client:libmozilla_voice_stt.so
//native_client:generate_scorer_package
"

if [ "${package_option}" = "--cuda" ]; then
    BAZEL_ENV_FLAGS="TF_NEED_CUDA=1 ${TF_CUDA_FLAGS}"
    BAZEL_BUILD_FLAGS="${BAZEL_CUDA_FLAGS} ${BAZEL_EXTRA_FLAGS} ${BAZEL_OPT_FLAGS}"
    PROJECT_NAME="Mozilla-Voice-STT-GPU"
elif [ "${package_option}" = "--tflite" ]; then
    PROJECT_NAME="Mozilla-Voice-STT-TFLite"
    BAZEL_BUILD_FLAGS="--define=runtime=tflite ${BAZEL_OPT_FLAGS} ${BAZEL_EXTRA_FLAGS}"
    BAZEL_ENV_FLAGS="TF_NEED_CUDA=0"
else
    PROJECT_NAME="Mozilla-Voice-STT"
    BAZEL_BUILD_FLAGS="${BAZEL_OPT_FLAGS} ${BAZEL_EXTRA_FLAGS}"
    BAZEL_ENV_FLAGS="TF_NEED_CUDA=0"
fi

SYSTEM_TARGET=host-win

do_bazel_build

if [ "${package_option}" = "--cuda" ]; then
    cp ${DS_ROOT_TASK}/DeepSpeech/ds/tensorflow/bazel-bin/native_client/liblibmozilla_voice_stt.so.ifso ${DS_ROOT_TASK}/DeepSpeech/ds/tensorflow/bazel-bin/native_client/libmozilla_voice_stt.so.if.lib
fi

export PATH=$PATH:$(cygpath ${ChocolateyInstall})/bin:'/c/Program Files/nodejs/'

do_deepspeech_binary_build

if [ "${runtime}" = "tflite" ]; then
  do_deepspeech_python_build "--tflite"
else
  do_deepspeech_python_build "${package_option}"
fi

do_deepspeech_nodejs_build "${package_option}"

do_deepspeech_netframework_build

do_deepspeech_netframework_wpf_build

do_nuget_build "${PROJECT_NAME}"

shutdown_bazel
