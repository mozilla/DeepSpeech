#!/bin/bash

set -xe

export OS=$(uname)
if [ "${OS}" = "Linux" ]; then
    export DS_ROOT_TASK=${HOME}
    export PYENV_ROOT="${DS_ROOT_TASK}/pyenv-root"
    export DS_CPU_COUNT=$(nproc)
fi;

if [ "${OS}" = "${TC_MSYS_VERSION}" ]; then
    export DS_ROOT_TASK=${TASKCLUSTER_TASK_DIR}
    export PYENV_ROOT="${TASKCLUSTER_TASK_DIR}/pyenv-root"
    export PLATFORM_EXE_SUFFIX=.exe
    export DS_CPU_COUNT=$(nproc)

    # Those are the versions available on NuGet.org
    export SUPPORTED_PYTHON_VERSIONS="3.5.4:ucs2 3.6.8:ucs2 3.7.6:ucs2 3.8.1:ucs2"
fi;

if [ "${OS}" = "Darwin" ]; then
    export DS_ROOT_TASK=${TASKCLUSTER_TASK_DIR}
    export DS_CPU_COUNT=$(sysctl hw.ncpu |cut -d' ' -f2)
    export PYENV_ROOT="${DS_ROOT_TASK}/pyenv-root"

    export HOMEBREW_NO_AUTO_UPDATE=1
    export BREW_URL=https://github.com/Homebrew/brew/tarball/2.2.17

    export BUILDS_BREW="${TASKCLUSTER_TASK_DIR}/homebrew-builds"
    export TESTS_BREW="${TASKCLUSTER_TASK_DIR}/homebrew-tests"

    export NVM_DIR=$TESTS_BREW/.nvm/ && mkdir -p $NVM_DIR
    export PKG_CONFIG_PATH="${BUILDS_BREW}/lib/pkgconfig"

    if [ -f "${BUILDS_BREW}/bin/brew" ]; then
        export PATH=${BUILDS_BREW}/bin/:${BUILDS_BREW}/opt/node@12/bin:$PATH
    fi;

    if [ -f "${TESTS_BREW}/bin/brew" ]; then
        export PATH=${TESTS_BREW}/bin/:$PATH
    fi;
fi;

export TASKCLUSTER_ARTIFACTS=${TASKCLUSTER_ARTIFACTS:-/tmp/artifacts}
export TASKCLUSTER_TMP_DIR=${TASKCLUSTER_TMP_DIR:-/tmp}

export ANDROID_TMP_DIR=/data/local/tmp

mkdir -p ${TASKCLUSTER_TMP_DIR} || true

export DS_TFDIR=${DS_ROOT_TASK}/DeepSpeech/ds/tensorflow
export DS_DSDIR=${DS_ROOT_TASK}/DeepSpeech/ds
export DS_EXAMPLEDIR=${DS_ROOT_TASK}/DeepSpeech/examples

export DS_VERSION="$(cat ${DS_DSDIR}/training/mozilla_voice_stt_training/VERSION)"

export GRADLE_USER_HOME=${DS_ROOT_TASK}/gradle-cache
export ANDROID_SDK_HOME=${DS_ROOT_TASK}/DeepSpeech/Android/SDK/
export ANDROID_NDK_HOME=${DS_ROOT_TASK}/DeepSpeech/Android/android-ndk-r18b/

WGET=${WGET:-"wget"}
TAR=${TAR:-"tar"}
XZ=${XZ:-"pixz -9"}
UNXZ=${UNXZ:-"pixz -d"}

if [ "${OS}" = "${TC_MSYS_VERSION}" ]; then
  WGET=/usr/bin/wget.exe
  TAR=/usr/bin/tar.exe
  XZ="xz -9 -T0 -c -"
  UNXZ="xz -9 -T0 -d"
fi

model_source="${DEEPSPEECH_TEST_MODEL}"
model_name="$(basename "${model_source}")"
model_name_mmap="$(basename -s ".pb" "${model_source}").pbmm"
model_source_mmap="$(dirname "${model_source}")/${model_name_mmap}"

ldc93s1_sample_filename=''

SUPPORTED_PYTHON_VERSIONS=${SUPPORTED_PYTHON_VERSIONS:-3.5.8:ucs2 3.6.10:ucs2 3.7.6:ucs2 3.8.1:ucs2}

# When updating NodeJS / ElectronJS supported versions, do not forget to increment
# deepspeech.node-gyp-cache.<X> in both `system.node_gyp_cache` (taskcluster/.shared.yml)
# and route index (taskcluster/node-gyp-cache.yml) to ensure the cache is updated
#
# Also, builds should always target first major branch release to ensure proper binary compatibility
# and tests should always target latest major branch release version
SUPPORTED_NODEJS_BUILD_VERSIONS=${SUPPORTED_NODEJS_BUILD_VERSIONS:-10.0.0 11.0.0 12.7.0 13.0.0 14.0.0}
SUPPORTED_NODEJS_TESTS_VERSIONS=${SUPPORTED_NODEJS_TESTS_VERSIONS:-10.20.1 11.15.0 12.17.0 13.14.0 14.3.0}

SUPPORTED_ELECTRONJS_VERSIONS=${SUPPORTED_ELECTRONJS_VERSIONS:-5.0.13 6.0.12 6.1.7 7.0.1 7.1.8 8.0.1 9.0.1 9.1.0 9.2.0}
