#!/bin/bash

set -xe

target=$1

source $(dirname $0)/tc-all-vars.sh

MAKE_TARGETS="lmplz filter build_binary"
CMAKE_BUILD="cmake --build . -j ${DS_CPU_COUNT} --target ${MAKE_TARGETS}"
CMAKE_DEFINES="-DFORCE_STATIC=ON"
case "${target}" in
    --android-arm64)
        export Boost_DIR="${DS_ROOT_TASK}/DeepSpeech/ds/ndk_21_boost_1.72.0/libs/arm64-v8a/cmake/Boost-1.72.0/"
        CMAKE_DEFINES="$CMAKE_DEFINES -DCMAKE_SYSTEM_NAME=Android"
        CMAKE_DEFINES="$CMAKE_DEFINES -DCMAKE_SYSTEM_VERSION=21"
        CMAKE_DEFINES="$CMAKE_DEFINES -DCMAKE_ANDROID_ARCH_ABI=arm64-v8a"
        CMAKE_DEFINES="$CMAKE_DEFINES -DCMAKE_ANDROID_NDK=${ANDROID_NDK_HOME}"
        CMAKE_DEFINES="$CMAKE_DEFINES -DCMAKE_ANDROID_STL_TYPE=c++_static"
        CMAKE_DEFINES="$CMAKE_DEFINES -DTHREADS_PTHREAD_ARG=2"
    ;;

    --android-armv7)
        export Boost_DIR="${DS_ROOT_TASK}/DeepSpeech/ds/ndk_21_boost_1.72.0/libs/armeabi-v7a/cmake/Boost-1.72.0/"
        CMAKE_DEFINES="$CMAKE_DEFINES -DCMAKE_SYSTEM_NAME=Android"
        CMAKE_DEFINES="$CMAKE_DEFINES -DCMAKE_SYSTEM_VERSION=21"
        CMAKE_DEFINES="$CMAKE_DEFINES -DCMAKE_ANDROID_ARCH_ABI=armeabi-v7a"
        CMAKE_DEFINES="$CMAKE_DEFINES -DCMAKE_ANDROID_NDK=${ANDROID_NDK_HOME}"
        CMAKE_DEFINES="$CMAKE_DEFINES -DCMAKE_ANDROID_STL_TYPE=c++_static"
        CMAKE_DEFINES="$CMAKE_DEFINES -DTHREADS_PTHREAD_ARG=2"
    ;;

    --android-x86_64)
        export Boost_DIR="${DS_ROOT_TASK}/DeepSpeech/ds/ndk_21_boost_1.72.0/libs/x86_64/cmake/Boost-1.72.0/"
        CMAKE_DEFINES="$CMAKE_DEFINES -DCMAKE_SYSTEM_NAME=Android"
        CMAKE_DEFINES="$CMAKE_DEFINES -DCMAKE_SYSTEM_VERSION=21"
        CMAKE_DEFINES="$CMAKE_DEFINES -DCMAKE_ANDROID_ARCH_ABI=x86_64"
        CMAKE_DEFINES="$CMAKE_DEFINES -DCMAKE_ANDROID_NDK=${ANDROID_NDK_HOME}"
        CMAKE_DEFINES="$CMAKE_DEFINES -DCMAKE_ANDROID_STL_TYPE=c++_static"
        CMAKE_DEFINES="$CMAKE_DEFINES -DTHREADS_PTHREAD_ARG=2"
    ;;

    --macos-amd64)
        export KENLM_BREW="${TASKCLUSTER_ORIG_TASKDIR}/homebrew-kenlm"
        export PATH=${KENLM_BREW}/bin:$PATH
        CMAKE_DEFINES="$CMAKE_DEFINES -DZLIB_LIBRARY=${KENLM_BREW}/opt/zlib/lib/libz.a"
        #CMAKE_DEFINES="$CMAKE_DEFINES -DBZIP2_LIBRARIES=${KENLM_BREW}/opt/bzip2/lib/libz2.a"
        CMAKE_DEFINES="$CMAKE_DEFINES -DTHREADS_PTHREAD_ARG=2"
    ;;

    --linux-arm64)
        TOOLCHAIN=${DS_ROOT_TASK}/DeepSpeech/ds/gcc-linaro-7.2.1-2017.11-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu
        CMAKE_DEFINES="$CMAKE_DEFINES -DCMAKE_SYSTEM_NAME=Linux"
        CMAKE_DEFINES="$CMAKE_DEFINES -DCMAKE_SYSTEM_PROCESSOR=aarch64"
        CMAKE_DEFINES="$CMAKE_DEFINES -DCMAKE_C_COMPILER=${TOOLCHAIN}-gcc"
        CMAKE_DEFINES="$CMAKE_DEFINES -DCMAKE_CXX_COMPILER=${TOOLCHAIN}-g++"
        CMAKE_DEFINES="$CMAKE_DEFINES -DCMAKE_SYSROOT=${DS_ROOT_TASK}/DeepSpeech/ds/multistrap-armbian64-buster/"
        CMAKE_DEFINES="$CMAKE_DEFINES -DTHREADS_PTHREAD_ARG=2"
        CMAKE_BUILD="make -j ${DS_CPU_COUNT} ${MAKE_TARGETS}"
    ;;

    --linux-rpi3)
        TOOLCHAIN=${DS_ROOT_TASK}/DeepSpeech/ds/gcc-linaro-7.2.1-2017.11-x86_64_arm-linux-gnueabihf/bin/arm-linux-gnueabihf
        CMAKE_DEFINES="$CMAKE_DEFINES -DCMAKE_SYSTEM_NAME=Linux"
        CMAKE_DEFINES="$CMAKE_DEFINES -DCMAKE_SYSTEM_PROCESSOR=arm"
        CMAKE_DEFINES="$CMAKE_DEFINES -DCMAKE_C_COMPILER=${TOOLCHAIN}-gcc"
        CMAKE_DEFINES="$CMAKE_DEFINES -DCMAKE_CXX_COMPILER=${TOOLCHAIN}-g++"
        CMAKE_DEFINES="$CMAKE_DEFINES -DCMAKE_SYSROOT=${DS_ROOT_TASK}/DeepSpeech/ds/multistrap-raspbian-buster/"
        CMAKE_DEFINES="$CMAKE_DEFINES -DTHREADS_PTHREAD_ARG=2"
        CMAKE_BUILD="make -j ${DS_CPU_COUNT} ${MAKE_TARGETS}"
    ;;

    --linux-amd64)
        CMAKE_BUILD="make -j ${DS_CPU_COUNT} ${MAKE_TARGETS}"
    ;;

    --windows-amd64)
        export Boost_DIR="$TASKCLUSTER_TASK_DIR/boost_1_72_0/lib64-msvc-14.2/cmake/Boost-1.72.0/"
        export CMAKE_GENERATOR='Visual Studio 16 2019'
        export PATH=$TASKCLUSTER_TASK_DIR/cmake-3.18.2-win64-x64/bin/:$PATH
        CMAKE_DEFINES="$CMAKE_DEFINES -A x64"
        CMAKE_DEFINES="$CMAKE_DEFINES -DLIBLZMA_LIBRARY=$TASKCLUSTER_TASK_DIR/xz-5.2.5/bin_x86-64/liblzma.a -DLIBLZMA_INCLUDE_DIR=$TASKCLUSTER_TASK_DIR/xz-5.2.5/include/"
        #CMAKE_DEFINES="$CMAKE_DEFINES -DBZIP2_LIBRARIES=$TASKCLUSTER_TASK_DIR/bzip2-dev-1.0.8.0-win-x64/libbz2-static.lib -DBZIP2_INCLUDE_DIR=$TASKCLUSTER_TASK_DIR/bzip2-dev-1.0.8.0-win-x64/"
    ;;
esac

mkdir ${DS_ROOT_TASK}/DeepSpeech/ds/kenlm/build/

pushd ${DS_ROOT_TASK}/DeepSpeech/ds/kenlm/build/
    export EIGEN3_ROOT=${DS_ROOT_TASK}/DeepSpeech/ds/eigen-3.3.7
    cmake ${CMAKE_DEFINES} ../
    ${CMAKE_BUILD}
popd
