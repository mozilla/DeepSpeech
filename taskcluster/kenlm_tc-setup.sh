#!/bin/bash

set -ex

target=$1

source $(dirname $0)/tc-all-vars.sh

mkdir -p ${TASKCLUSTER_ARTIFACTS} || true

pushd ${DS_ROOT_TASK}/DeepSpeech/ds/
    case "${target}" in
        --android-arm64|--android-armv7|--android-x86_64)
            source $(dirname $0)/tc-android-utils.sh

            wget -q https://github.com/dec1/Boost-for-Android/releases/download/ndk_21_boost_1.72.0/ndk_21_boost_1.72.0.zip && unzip ndk_21_boost_1.72.0.zip
            mkdir -p ${ANDROID_NDK_HOME} || true
            android_install_ndk
        ;;

        --macos-amd64)
            source $(dirname $0)/homebrew-build.sh
            export KENLM_BREW="${TASKCLUSTER_ORIG_TASKDIR}/homebrew-kenlm"
            do_prepare_homebrew "${KENLM_BREW}"

            install_pkg_homebrew "cmake"
            install_pkg_homebrew "coreutils"
            install_pkg_homebrew "boost"
            install_pkg_homebrew "bzip2"
            install_pkg_homebrew "zlib"
            install_pkg_homebrew "xz"
        ;;

        --linux-arm64)
            wget -q -O - https://releases.linaro.org/components/toolchain/binaries/7.2-2017.11/aarch64-linux-gnu/gcc-linaro-7.2.1-2017.11-x86_64_aarch64-linux-gnu.tar.xz | pixz -d | tar -xf -
            multistrap -d multistrap-armbian64-buster/ -f taskcluster/kenlm_multistrap_arm64_buster.conf

            # fix all symlink
            ls -hal multistrap-armbian64-buster/usr/lib/aarch64-linux-gnu/libpthread.so
            ls -hal multistrap-armbian64-buster/usr/lib/aarch64-linux-gnu/libz.so
            ls -hal multistrap-armbian64-buster/usr/lib/aarch64-linux-gnu/libbz2.so
            ls -hal multistrap-armbian64-buster/usr/lib/aarch64-linux-gnu/libm.so

            rm multistrap-armbian64-buster/usr/lib/aarch64-linux-gnu/libpthread.so
            rm multistrap-armbian64-buster/usr/lib/aarch64-linux-gnu/libz.so
            rm multistrap-armbian64-buster/usr/lib/aarch64-linux-gnu/libbz2.so
            rm multistrap-armbian64-buster/usr/lib/aarch64-linux-gnu/libm.so

            ln -s ../../../lib/aarch64-linux-gnu/libpthread.so.0 multistrap-armbian64-buster/usr/lib/aarch64-linux-gnu/libpthread.so
            ln -s ../../../lib/aarch64-linux-gnu/libz.so.1.2.11 multistrap-armbian64-buster/usr/lib/aarch64-linux-gnu/libz.so
            ln -s ../../../lib/aarch64-linux-gnu/libbz2.so.1.0 multistrap-armbian64-buster/usr/lib/aarch64-linux-gnu/libbz2.so
            ln -s ../../../lib/aarch64-linux-gnu/libm.so.6 multistrap-armbian64-buster/usr/lib/aarch64-linux-gnu/libm.so

            ls -hal multistrap-armbian64-buster/usr/lib/aarch64-linux-gnu/libpthread.so
            ls -hal multistrap-armbian64-buster/usr/lib/aarch64-linux-gnu/libz.so
            ls -hal multistrap-armbian64-buster/usr/lib/aarch64-linux-gnu/libbz2.so
            ls -hal multistrap-armbian64-buster/usr/lib/aarch64-linux-gnu/libm.so
        ;;

        --linux-rpi3)
            wget -q -O - https://releases.linaro.org/components/toolchain/binaries/7.2-2017.11/arm-linux-gnueabihf/gcc-linaro-7.2.1-2017.11-x86_64_arm-linux-gnueabihf.tar.xz | pixz -d | tar -xf -
            multistrap -d multistrap-raspbian-buster/ -f taskcluster/kenlm_multistrap_rpi3_buster.conf

            # fix all symlink
            ls -hal multistrap-raspbian-buster/usr/lib/arm-linux-gnueabihf/libpthread.so
            ls -hal multistrap-raspbian-buster/usr/lib/arm-linux-gnueabihf/libz.so
            ls -hal multistrap-raspbian-buster/usr/lib/arm-linux-gnueabihf/libbz2.so
            ls -hal multistrap-raspbian-buster/usr/lib/arm-linux-gnueabihf/libm.so

            rm multistrap-raspbian-buster/usr/lib/arm-linux-gnueabihf/libpthread.so
            rm multistrap-raspbian-buster/usr/lib/arm-linux-gnueabihf/libz.so
            rm multistrap-raspbian-buster/usr/lib/arm-linux-gnueabihf/libbz2.so
            rm multistrap-raspbian-buster/usr/lib/arm-linux-gnueabihf/libm.so

            ln -s ../../../lib/arm-linux-gnueabihf/libpthread.so.0 multistrap-raspbian-buster/usr/lib/arm-linux-gnueabihf/libpthread.so
            ln -s ../../../lib/arm-linux-gnueabihf/libz.so.1.2.11 multistrap-raspbian-buster/usr/lib/arm-linux-gnueabihf/libz.so
            ln -s ../../../lib/arm-linux-gnueabihf/libbz2.so.1.0 multistrap-raspbian-buster/usr/lib/arm-linux-gnueabihf/libbz2.so
            ln -s ../../../lib/arm-linux-gnueabihf/libm.so.6 multistrap-raspbian-buster/usr/lib/arm-linux-gnueabihf/libm.so

            ls -hal multistrap-raspbian-buster/usr/lib/arm-linux-gnueabihf/libpthread.so
            ls -hal multistrap-raspbian-buster/usr/lib/arm-linux-gnueabihf/libz.so
            ls -hal multistrap-raspbian-buster/usr/lib/arm-linux-gnueabihf/libbz2.so
            ls -hal multistrap-raspbian-buster/usr/lib/arm-linux-gnueabihf/libm.so
        ;;

        --windows-amd64)
            wget -q https://github.com/Kitware/CMake/releases/download/v3.18.2/cmake-3.18.2-win64-x64.zip -O cmake-3.18.2-win64-x64.zip && \
            "C:\Program Files\7-zip\7z.exe" x -o$TASKCLUSTER_TASK_DIR/ -tzip -aoa cmake-3.18.2-win64-x64.zip;
            rm cmake-*.zip

            wget -q https://bintray.com/boostorg/release/download_file?file_path=1.72.0%2Fbinaries%2Fboost_1_72_0-bin-msvc-all-32-64.7z -O boost_1_72_0-bin-msvc-all-32-64.7z && \
            "C:\Program Files\7-zip\7z.exe" x -o$TASKCLUSTER_TASK_DIR/ -t7z -aoa boost_1_72_0-bin-msvc-all-32-64.7z;
            rm boost_*.7z

            wget -q https://tukaani.org/xz/xz-5.2.5-windows.7z -O xz-5.2.5-windows.7z && \
            "C:\Program Files\7-zip\7z.exe" x -o$TASKCLUSTER_TASK_DIR/xz-5.2.5/ -t7z -aoa xz-5.2.5-windows.7z;
            rm xz-*.7z

            wget -q https://github.com/philr/bzip2-windows/releases/download/v1.0.8.0/bzip2-dev-1.0.8.0-win-x64.zip -O bzip2-dev-1.0.8.0-win-x64.zip && \
            "C:\Program Files\7-zip\7z.exe" x -o$TASKCLUSTER_TASK_DIR/bzip2-dev-1.0.8.0-win-x64/ -tzip -aoa bzip2-dev-1.0.8.0-win-x64.zip;
            rm bzip2-dev-*.zip
        ;;
    esac

    git submodule --quiet sync kenlm/ && git submodule --quiet update --init kenlm/

    export EIGEN3_ROOT=${DS_ROOT_TASK}/DeepSpeech/ds/eigen-3.3.7
    wget -q -O - https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.tar.bz2 |tar xj
popd
