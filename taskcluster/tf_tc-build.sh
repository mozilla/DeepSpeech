#!/bin/bash

set -ex

source $(dirname $0)/tf_tc-vars.sh

pushd ${DS_ROOT_TASK}/DeepSpeech/ds/tensorflow/
    BAZEL_BUILD="bazel ${BAZEL_OUTPUT_USER_ROOT} build -s --explain bazel_monolithic_tf.log --verbose_explanations --experimental_strict_action_env --config=monolithic"

    # Start a bazel process to ensure reliability on Windows and avoid:
    # FATAL: corrupt installation: file 'c:\builds\tc-workdir\.bazel_cache/install/6b1660721930e9d5f231f7d2a626209b/_embedded_binaries/build-runfiles.exe' missing.
    bazel ${BAZEL_OUTPUT_USER_ROOT} info

    # Force toolchain sync (useful on macOS ?)
    bazel ${BAZEL_OUTPUT_USER_ROOT} sync --configure

    case "$1" in
    "--cpu")
        echo "" | TF_NEED_CUDA=0 ./configure && ${BAZEL_BUILD} -c opt ${BAZEL_OPT_FLAGS} ${BAZEL_EXTRA_FLAGS} ${BUILD_TARGET_LIB_CPP_API} ${BUILD_TARGET_LITE_LIB}
        ;;
    "--gpu")
        eval "export ${TF_CUDA_FLAGS}" && (echo "" | TF_NEED_CUDA=1 ./configure) && ${BAZEL_BUILD} -c opt ${BAZEL_CUDA_FLAGS} ${BAZEL_EXTRA_FLAGS} ${BAZEL_OPT_FLAGS} ${BUILD_TARGET_LIB_CPP_API}
        ;;
    "--arm")
        echo "" | TF_NEED_CUDA=0 ./configure && ${BAZEL_BUILD} -c opt ${BAZEL_ARM_FLAGS} ${BAZEL_EXTRA_FLAGS} ${BUILD_TARGET_LITE_LIB}
        ;;
    "--arm64")
        echo "" | TF_NEED_CUDA=0 ./configure && ${BAZEL_BUILD} -c opt ${BAZEL_ARM64_FLAGS} ${BAZEL_EXTRA_FLAGS} ${BUILD_TARGET_LITE_LIB}
        ;;
    "--android-armv7")
        echo "" | TF_SET_ANDROID_WORKSPACE=1 ./configure && ${BAZEL_BUILD} -c opt ${BAZEL_ANDROID_ARM_FLAGS} ${BAZEL_EXTRA_FLAGS} ${BUILD_TARGET_LITE_LIB}
        ;;
    "--android-arm64")
        echo "" | TF_SET_ANDROID_WORKSPACE=1 ./configure && ${BAZEL_BUILD} -c opt ${BAZEL_ANDROID_ARM64_FLAGS} ${BAZEL_EXTRA_FLAGS} ${BUILD_TARGET_LITE_LIB}
        ;;
    "--ios-arm64")
        echo "" | TF_NEED_CUDA=0 TF_CONFIGURE_IOS=1 ./configure && ${BAZEL_BUILD} -c opt ${BAZEL_IOS_ARM64_FLAGS} ${BUILD_TARGET_LITE_LIB}
        ;;
    "--ios-x86_64")
        echo "" | TF_NEED_CUDA=0 TF_CONFIGURE_IOS=1 ./configure && ${BAZEL_BUILD} -c opt ${BAZEL_IOS_X86_64_FLAGS} ${BUILD_TARGET_LITE_LIB}
        ;;
    esac

    bazel ${BAZEL_OUTPUT_USER_ROOT} shutdown
popd
