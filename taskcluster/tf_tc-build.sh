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

    OPT_OR_DBG=${2:-opt}

    case "$1" in
    "--linux-cpu"|"--darwin-cpu"|"--windows-cpu")
        echo "" | TF_NEED_CUDA=0 ./configure && ${BAZEL_BUILD} -c ${OPT_OR_DBG} ${BAZEL_OPT_FLAGS} ${BAZEL_EXTRA_FLAGS} ${BUILD_TARGET_LIB_CPP_API} ${BUILD_TARGET_LITE_LIB}
        ;;
    "--linux-cuda"|"--windows-cuda")
        eval "export ${TF_CUDA_FLAGS}" && (echo "" | TF_NEED_CUDA=1 ./configure) && ${BAZEL_BUILD} -c ${OPT_OR_DBG} ${BAZEL_CUDA_FLAGS} ${BAZEL_EXTRA_FLAGS} ${BAZEL_OPT_FLAGS} ${BUILD_TARGET_LIB_CPP_API}
        ;;
    "--linux-arm")
        echo "" | TF_NEED_CUDA=0 ./configure && ${BAZEL_BUILD} -c ${OPT_OR_DBG} ${BAZEL_ARM_FLAGS} ${BAZEL_EXTRA_FLAGS} ${BUILD_TARGET_LITE_LIB}
        ;;
    "--linux-arm64")
        echo "" | TF_NEED_CUDA=0 ./configure && ${BAZEL_BUILD} -c ${OPT_OR_DBG} ${BAZEL_ARM64_FLAGS} ${BAZEL_EXTRA_FLAGS} ${BUILD_TARGET_LITE_LIB}
        ;;
    "--android-armv7")
        echo "" | TF_SET_ANDROID_WORKSPACE=1 ./configure && ${BAZEL_BUILD} -c ${OPT_OR_DBG} ${BAZEL_ANDROID_ARM_FLAGS} ${BAZEL_EXTRA_FLAGS} ${BUILD_TARGET_LITE_LIB}
        ;;
    "--android-arm64")
        echo "" | TF_SET_ANDROID_WORKSPACE=1 ./configure && ${BAZEL_BUILD} -c ${OPT_OR_DBG} ${BAZEL_ANDROID_ARM64_FLAGS} ${BAZEL_EXTRA_FLAGS} ${BUILD_TARGET_LITE_LIB}
        ;;
    "--ios-arm64")
        echo "" | TF_NEED_CUDA=0 TF_CONFIGURE_IOS=1 ./configure && ${BAZEL_BUILD} -c ${OPT_OR_DBG} ${BAZEL_IOS_ARM64_FLAGS} ${BUILD_TARGET_LITE_LIB}
        ;;
    "--ios-x86_64")
        echo "" | TF_NEED_CUDA=0 TF_CONFIGURE_IOS=1 ./configure && ${BAZEL_BUILD} -c ${OPT_OR_DBG} ${BAZEL_IOS_X86_64_FLAGS} ${BUILD_TARGET_LITE_LIB}
        ;;
    esac

    bazel ${BAZEL_OUTPUT_USER_ROOT} shutdown
popd
