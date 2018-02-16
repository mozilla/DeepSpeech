#!/bin/bash

set -xe

source $(dirname "$0")/../tc-tests-utils.sh

source ${DS_ROOT_TASK}/DeepSpeech/tf/tc-vars.sh

BAZEL_TARGETS="
//native_client:libdeepspeech.so
//native_client:deepspeech_utils
//native_client:generate_trie
"

BAZEL_ENV_FLAGS="TF_NEED_OPENCL_SYCL=1 ${TF_CCPP_FLAGS}"
BAZEL_BUILD_FLAGS="${BAZEL_CCPP_FLAGS} ${BAZEL_EXTRA_FLAGS} ${BAZEL_OPT_FLAGS}"
SYSTEM_TARGET=host
EXTRA_LOCAL_CFLAGS="-Wl,-rpath-link,${DS_ROOT_TASK}/DeepSpeech/ComputeCpp-CE/lib/"
EXTRA_LOCAL_LDFLAGS=""

do_bazel_build

do_deepspeech_binary_build

do_deepspeech_python_build "gpu-opencl"

do_deepspeech_nodejs_build "gpu-opencl"

$(dirname "$0")/decoder-build.sh
