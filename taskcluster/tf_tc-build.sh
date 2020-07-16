#!/bin/bash

set -ex

source $(dirname $0)/tf_tc-vars.sh

build_amd64=no
build_gpu=no
build_android_arm=no
build_android_arm64=no
build_linux_arm=no
build_linux_arm64=no
build_ios_arm64=no
build_ios_x86_64=no

if [ "$1" = "--cpu" ]; then
    build_amd64=yes
fi

if [ "$1" = "--gpu" ]; then
    build_amd64=yes
    build_gpu=yes
fi

if [ "$1" = "--arm" ]; then
    build_amd64=yes
    build_linux_arm=yes
fi

if [ "$1" = "--arm64" ]; then
    build_amd64=yes
    build_linux_arm64=yes
fi

if [ "$1" = "--android-armv7" ]; then
    build_android_arm=yes
fi

if [ "$1" = "--android-arm64" ]; then
    build_android_arm64=yes
fi

if [ "$1" = "--ios-arm64" ]; then
    build_ios_arm64=yes
fi

if [ "$1" = "--ios-x86_64" ]; then
    build_ios_x86_64=yes
fi

pushd ${DS_ROOT_TASK}/DeepSpeech/ds/tensorflow/
    BAZEL_BUILD="bazel ${BAZEL_OUTPUT_USER_ROOT} build -s --explain bazel_monolithic_tf.log --verbose_explanations --experimental_strict_action_env --config=monolithic"

    # Start a bazel process to ensure reliability on Windows and avoid:
    # FATAL: corrupt installation: file 'c:\builds\tc-workdir\.bazel_cache/install/6b1660721930e9d5f231f7d2a626209b/_embedded_binaries/build-runfiles.exe' missing.
    bazel ${BAZEL_OUTPUT_USER_ROOT} info

    # Force toolchain sync (useful on macOS ?)
    bazel ${BAZEL_OUTPUT_USER_ROOT} sync --configure

    if [ "${build_amd64}" = "yes" ]; then
        # Pure amd64 CPU-only build
        if [ "${OS}" = "${TC_MSYS_VERSION}" -a  "${build_gpu}" = "no" ]; then
            echo "" | TF_NEED_CUDA=0 ./configure && ${BAZEL_BUILD} -c opt ${BAZEL_OPT_FLAGS} ${BAZEL_EXTRA_FLAGS} ${BUILD_TARGET_LIB_CPP_API} ${BUILD_TARGET_LITE_LIB}
        elif [ "${build_gpu}" = "no" -a "${build_linux_arm}" = "no" -a "${build_linux_arm64}" = "no" ]; then
            echo "" | TF_NEED_CUDA=0 ./configure && ${BAZEL_BUILD} -c opt ${BAZEL_OPT_FLAGS} ${BAZEL_EXTRA_FLAGS} ${BUILD_TARGET_LIB_CPP_API} ${BUILD_TARGET_LITE_LIB}
        fi

        # Cross RPi3 CPU-only build
        if [ "${build_linux_arm}" = "yes" ]; then
            echo "" | TF_NEED_CUDA=0 ./configure && ${BAZEL_BUILD} -c opt ${BAZEL_ARM_FLAGS} ${BAZEL_EXTRA_FLAGS} ${BUILD_TARGET_LITE_LIB}
        fi

        # Cross ARM64 Cortex-A53 build
        if [ "${build_linux_arm64}" = "yes" ]; then
            echo "" | TF_NEED_CUDA=0 ./configure && ${BAZEL_BUILD} -c opt ${BAZEL_ARM64_FLAGS} ${BAZEL_EXTRA_FLAGS} ${BUILD_TARGET_LITE_LIB}
        fi

        # Pure amd64 GPU-enabled build
        if [ "${build_gpu}" = "yes" ]; then
            eval "export ${TF_CUDA_FLAGS}" && (echo "" | TF_NEED_CUDA=1 ./configure) && ${BAZEL_BUILD} -c opt ${BAZEL_CUDA_FLAGS} ${BAZEL_EXTRA_FLAGS} ${BAZEL_OPT_FLAGS} ${BUILD_TARGET_LIB_CPP_API}
        fi
    fi

    if [ "${build_android_arm}" = "yes" ]; then
        echo "" | TF_SET_ANDROID_WORKSPACE=1 ./configure && ${BAZEL_BUILD} -c opt ${BAZEL_ANDROID_ARM_FLAGS} ${BAZEL_EXTRA_FLAGS} ${BUILD_TARGET_LITE_LIB}
    fi;

    if [ "${build_android_arm64}" = "yes" ]; then
        echo "" | TF_SET_ANDROID_WORKSPACE=1 ./configure && ${BAZEL_BUILD} -c opt ${BAZEL_ANDROID_ARM64_FLAGS} ${BAZEL_EXTRA_FLAGS} ${BUILD_TARGET_LITE_LIB}
    fi;

    if [ "${build_ios_arm64}" = "yes" ]; then
        echo "" | TF_NEED_CUDA=0 TF_CONFIGURE_IOS=1 ./configure && ${BAZEL_BUILD} -c opt ${BAZEL_IOS_ARM64_FLAGS} ${BUILD_TARGET_LITE_LIB}
    fi;

    if [ "${build_ios_x86_64}" = "yes" ]; then
        echo "" | TF_NEED_CUDA=0 TF_CONFIGURE_IOS=1 ./configure && ${BAZEL_BUILD} -c opt ${BAZEL_IOS_X86_64_FLAGS} ${BUILD_TARGET_LITE_LIB}
    fi;

    if [ $? -ne 0 ]; then
        # There was a failure, just account for it.
        echo "Build failure, please check the output above. Exit code was: $?"
        return 1
    fi

    bazel ${BAZEL_OUTPUT_USER_ROOT} shutdown
popd
