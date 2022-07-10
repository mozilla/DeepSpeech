#!/bin/bash

set -ex

export OS=$(uname)
if [ "${OS}" = "Linux" ]; then
    export DS_ROOT_TASK=${CI_TASK_DIR}

    BAZEL_URL=https://github.com/bazelbuild/bazel/releases/download/3.1.0/bazel-3.1.0-installer-linux-x86_64.sh
    BAZEL_SHA256=7ba815cbac712d061fe728fef958651512ff394b2708e89f79586ec93d1185ed

    CUDA_URL=http://developer.download.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda_10.1.243_418.87.00_linux.run
    CUDA_SHA256=e7c22dc21278eb1b82f34a60ad7640b41ad3943d929bebda3008b72536855d31

    # From https://gitlab.com/nvidia/cuda/blob/centos7/10.1/devel/cudnn7/Dockerfile
    CUDNN_URL=http://developer.download.nvidia.com/compute/redist/cudnn/v7.6.0/cudnn-10.1-linux-x64-v7.6.0.64.tgz
    CUDNN_SHA256=e956c6f9222fcb867a10449cfc76dee5cfd7c7531021d95fe9586d7e043b57d7

    ANDROID_NDK_URL=https://dl.google.com/android/repository/android-ndk-r18b-linux-x86_64.zip
    ANDROID_NDK_SHA256=4f61cbe4bbf6406aa5ef2ae871def78010eed6271af72de83f8bd0b07a9fd3fd

    ANDROID_SDK_URL=https://dl.google.com/android/repository/sdk-tools-linux-4333796.zip
    ANDROID_SDK_SHA256=92ffee5a1d98d856634e8b71132e8a95d96c83a63fde1099be3d86df3106def9

    WGET=/usr/bin/wget
elif [ "${OS}" = "${CI_MSYS_VERSION}" ]; then
    if [ -z "${CI_TASK_DIR}" -o -z "${CI_ARTIFACTS_DIR}" ]; then
        echo "Inconsistent Windows setup: missing some vars."
        echo "CI_TASK_DIR=${CI_TASK_DIR}"
        echo "CI_ARTIFACTS_DIR=${CI_ARTIFACTS_DIR}"
        exit 1
    fi;

    # Re-export with cygpath to make sure it is sane, otherwise it might trigger
    # unobvious failures with cp etc.
    export CI_TASK_DIR="$(cygpath ${CI_TASK_DIR})"
    export CI_ARTIFACTS_DIR="$(cygpath ${CI_ARTIFACTS_DIR})"

    export DS_ROOT_TASK=${CI_TASK_DIR}
    export BAZEL_VC="C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\VC"
    export BAZEL_VC_FULL_VERSION="14.28.29910"
    export MSYS2_ARG_CONV_EXCL='//'

    mkdir -p ${CI_TASK_DIR}/tmp/
    export TEMP=${CI_TASK_DIR}/tmp/
    export TMP=${CI_TASK_DIR}/tmp/

    BAZEL_URL=https://github.com/bazelbuild/bazel/releases/download/3.1.0/bazel-3.1.0-windows-x86_64.exe
    BAZEL_SHA256=776db1f4986dacc3eda143932f00f7529f9ee65c7c1c004414c44aaa6419d0e9

    CUDA_INSTALL_DIRECTORY=$(cygpath 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1')

    TAR=/usr/bin/tar.exe
elif [ "${OS}" = "Darwin" ]; then
    if [ -z "${CI_TASK_DIR}" -o -z "${CI_ARTIFACTS_DIR}" ]; then
        echo "Inconsistent OSX setup: missing some vars."
        echo "CI_TASK_DIR=${CI_TASK_DIR}"
        echo "CI_ARTIFACTS_DIR=${CI_ARTIFACTS_DIR}"
        exit 1
    fi;

    export DS_ROOT_TASK=${CI_TASK_DIR}

    BAZEL_URL=https://github.com/bazelbuild/bazel/releases/download/3.1.0/bazel-3.1.0-installer-darwin-x86_64.sh
    BAZEL_SHA256=5cfa97031b43432b3c742c80e2e01c41c0acdca7ba1052fc8cf1e291271bc9cd

    SHA_SUM="shasum -a 256 -c"
    TAR=gtar
fi;

WGET=${WGET:-"wget"}
TAR=${TAR:-"tar"}
XZ=${XZ:-"xz -9 -T0"}
ZIP=${ZIP:-"zip"}
UNXZ=${UNXZ:-"xz -T0 -d"}
UNGZ=${UNGZ:-"gunzip"}
SHA_SUM=${SHA_SUM:-"sha256sum -c --strict"}

# /tmp/artifacts for docker-worker on linux,
# and task subdir for generic-worker on osx
export CI_ARTIFACTS_DIR=${CI_ARTIFACTS_DIR:-/tmp/artifacts}

### Define variables that needs to be exported to other processes

PATH=${DS_ROOT_TASK}/bin:$PATH
if [ "${OS}" = "Darwin" ]; then
    PATH=${DS_ROOT_TASK}/homebrew/bin/:${DS_ROOT_TASK}/homebrew/opt/node@10/bin:$PATH
fi;
export PATH

if [ "${OS}" = "Linux" ]; then
    export LD_LIBRARY_PATH=${DS_ROOT_TASK}/DeepSpeech/CUDA/lib64/:${DS_ROOT_TASK}/DeepSpeech/CUDA/lib64/stubs/:$LD_LIBRARY_PATH
    export ANDROID_SDK_HOME=${DS_ROOT_TASK}/DeepSpeech/Android/SDK/
    export ANDROID_NDK_HOME=${DS_ROOT_TASK}/DeepSpeech/Android/android-ndk-r18b/
fi;

export TF_ENABLE_XLA=0
if [ "${OS}" = "Linux" ]; then
    TF_NEED_JEMALLOC=1
elif [ "${OS}" = "${CI_MSYS_VERSION}" ]; then
    TF_NEED_JEMALLOC=0
elif [ "${OS}" = "Darwin" ]; then
    TF_NEED_JEMALLOC=0
fi;
export TF_NEED_JEMALLOC
export TF_NEED_OPENCL_SYCL=0
export TF_NEED_MKL=0
export TF_NEED_VERBS=0
export TF_NEED_MPI=0
export TF_NEED_IGNITE=0
export TF_NEED_GDR=0
export TF_NEED_NGRAPH=0
export TF_DOWNLOAD_CLANG=0
export TF_SET_ANDROID_WORKSPACE=0
export TF_NEED_TENSORRT=0
export TF_NEED_ROCM=0

# This should be gcc-5, hopefully. CUDA and TensorFlow might not be happy, otherwise.
export GCC_HOST_COMPILER_PATH=/usr/bin/gcc

if [ "${OS}" = "Linux" ]; then
    source /etc/os-release
    if [ "${ID}" = "ubuntu" -a "${VERSION_ID}" = "20.04" ]; then
        export PYTHON_BIN_PATH=/usr/bin/python3
    else
        export PYTHON_BIN_PATH=/usr/bin/python2.7
    fi
fi

## Below, define or export some build variables

# Enable some SIMD support. Limit ourselves to what Tensorflow needs.
# Also ensure to not require too recent CPU: AVX2/FMA introduced by:
#  - Intel with Haswell (2013)
#  - AMD with Excavator (2015)
# For better compatibility, AVX ony might be better.
#
# Build for generic amd64 platforms, no device-specific optimization
# See https://gcc.gnu.org/onlinedocs/gcc/x86-Options.html for targetting specific CPUs

if [ "${OS}" = "${CI_MSYS_VERSION}" ]; then
    OPT_FLAGS="/arch:AVX"
else
    OPT_FLAGS="-mtune=generic -march=x86-64 -msse -msse2 -msse3 -msse4.1 -msse4.2 -mavx"
fi
BAZEL_OPT_FLAGS=""
for flag in ${OPT_FLAGS};
do
    BAZEL_OPT_FLAGS="${BAZEL_OPT_FLAGS} --copt=${flag}"
done;

BAZEL_OUTPUT_CACHE_DIR="${DS_ROOT_TASK}/.bazel_cache/"
BAZEL_OUTPUT_CACHE_INSTANCE="${BAZEL_OUTPUT_CACHE_DIR}/output/"
mkdir -p ${BAZEL_OUTPUT_CACHE_INSTANCE} || true

# We need both to ensure stable path ; default value for output_base is some
# MD5 value.
BAZEL_OUTPUT_USER_ROOT="--output_user_root ${BAZEL_OUTPUT_CACHE_DIR} --output_base ${BAZEL_OUTPUT_CACHE_INSTANCE}"
export BAZEL_OUTPUT_USER_ROOT

NVCC_COMPUTE="3.5"

### Define build parameters/env variables that we will re-ues in sourcing scripts.
if [ "${OS}" = "${CI_MSYS_VERSION}" ]; then
    TF_CUDA_FLAGS="TF_CUDA_CLANG=0 TF_CUDA_VERSION=10.1 TF_CUDNN_VERSION=7.6.0 CUDNN_INSTALL_PATH=\"${CUDA_INSTALL_DIRECTORY}\" TF_CUDA_PATHS=\"${CUDA_INSTALL_DIRECTORY}\" TF_CUDA_COMPUTE_CAPABILITIES=\"${NVCC_COMPUTE}\""
else
    TF_CUDA_FLAGS="TF_CUDA_CLANG=0 TF_CUDA_VERSION=10.1 TF_CUDNN_VERSION=7.6.0 CUDNN_INSTALL_PATH=\"${DS_ROOT_TASK}/DeepSpeech/CUDA\" TF_CUDA_PATHS=\"${DS_ROOT_TASK}/DeepSpeech/CUDA\" TF_CUDA_COMPUTE_CAPABILITIES=\"${NVCC_COMPUTE}\""
fi
BAZEL_ARM_FLAGS="--config=rpi3 --config=rpi3_opt --copt=-DTFLITE_WITH_RUY_GEMV"
BAZEL_ARM64_FLAGS="--config=rpi3-armv8 --config=rpi3-armv8_opt --copt=-DTFLITE_WITH_RUY_GEMV"
BAZEL_ANDROID_ARM_FLAGS="--config=android --config=android_arm --action_env ANDROID_NDK_API_LEVEL=21 --cxxopt=-std=c++14 --copt=-D_GLIBCXX_USE_C99 --copt=-DTFLITE_WITH_RUY_GEMV"
BAZEL_ANDROID_ARM64_FLAGS="--config=android --config=android_arm64 --action_env ANDROID_NDK_API_LEVEL=21 --cxxopt=-std=c++14 --copt=-D_GLIBCXX_USE_C99 --copt=-DTFLITE_WITH_RUY_GEMV"
BAZEL_CUDA_FLAGS="--config=cuda"
if [ "${OS}" = "Linux" ]; then
    # constexpr usage in tensorflow's absl dep fails badly because of gcc-5
    # so let's skip that
    BAZEL_CUDA_FLAGS="${BAZEL_CUDA_FLAGS} --copt=-DNO_CONSTEXPR_FOR_YOU=1"
fi
BAZEL_IOS_ARM64_FLAGS="--config=ios_arm64 --define=runtime=tflite --copt=-DTFLITE_WITH_RUY_GEMV"
BAZEL_IOS_X86_64_FLAGS="--config=ios_x86_64 --define=runtime=tflite --copt=-DTFLITE_WITH_RUY_GEMV"

if [ "${OS}" != "${CI_MSYS_VERSION}" ]; then
    BAZEL_EXTRA_FLAGS="--config=noaws --config=nogcp --config=nohdfs --config=nonccl --copt=-fvisibility=hidden"
fi

if [ "${OS}" = "Darwin" ]; then
    BAZEL_EXTRA_FLAGS="${BAZEL_EXTRA_FLAGS} --macos_minimum_os 10.10 --macos_sdk_version 10.15"
fi

### Define build targets that we will re-ues in sourcing scripts.
BUILD_TARGET_LIB_CPP_API="//tensorflow:tensorflow_cc"
BUILD_TARGET_GRAPH_TRANSFORMS="//tensorflow/tools/graph_transforms:transform_graph"
BUILD_TARGET_GRAPH_SUMMARIZE="//tensorflow/tools/graph_transforms:summarize_graph"
BUILD_TARGET_GRAPH_BENCHMARK="//tensorflow/tools/benchmark:benchmark_model"
#BUILD_TARGET_CONVERT_MMAP="//tensorflow/contrib/util:convert_graphdef_memmapped_format"
BUILD_TARGET_TOCO="//tensorflow/lite/toco:toco"
BUILD_TARGET_LITE_BENCHMARK="//tensorflow/lite/tools/benchmark:benchmark_model"
BUILD_TARGET_LITE_LIB="//tensorflow/lite/c:libtensorflowlite_c.so"
BUILD_TARGET_LIBDEEPSPEECH="//native_client:libdeepspeech.so"
