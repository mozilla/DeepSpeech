#!/bin/bash

set -xe

source ${HOME}/DeepSpeech/tf/tc-vars.sh

DS_TFDIR=${HOME}/DeepSpeech/tf
EXTRA_CUDA_CFLAGS=
EXTRA_CUDA_LDFLAGS=

if [ "$1" = "--gpu" ]; then
    BAZEL_ENV_FLAGS="TF_NEED_CUDA=1 ${TF_CUDA_FLAGS}"
    BAZEL_BUILD_FLAGS="${BAZEL_CUDA_FLAGS} ${BAZEL_OPT_FLAGS}"
    SYSTEM_TARGET=host
    EXTRA_CUDA_CFLAGS=-L${HOME}/DeepSpeech/CUDA/lib64/
    EXTRA_CUDA_LDFLAGS=-lcudart
fi

if [ "$1" = "--arm" ]; then
    BAZEL_ENV_FLAGS="TF_NEED_CUDA=0"
    BAZEL_BUILD_FLAGS="${BAZEL_ARM_FLAGS}"
    SYSTEM_TARGET=rpi3
fi

if [ "$1" != "--gpu" -a "$1" != "--arm" ]; then
    BAZEL_ENV_FLAGS="TF_NEED_CUDA=0"
    BAZEL_BUILD_FLAGS="${BAZEL_OPT_FLAGS}"
    SYSTEM_TARGET=host
    MAKE_BINDINGS=1
fi

cd ~/DeepSpeech/tf
eval "export ${BAZEL_ENV_FLAGS}"
PATH=${HOME}/bin/:$PATH bazel \
	build -c opt ${BAZEL_BUILD_FLAGS} \
	//native_client:*

cd ~/DeepSpeech/ds/
make -C native_client/ \
	TARGET=${SYSTEM_TARGET} \
	TFDIR=${DS_TFDIR} \
	RASPBIAN=/tmp/multistrap-raspbian-jessie \
	EXTRA_CFLAGS=${EXTRA_CUDA_CFLAGS} \
	EXTRA_LDFLAGS=${EXTRA_CUDA_LDFLAGS} \
	deepspeech

if [ ${MAKE_BINDINGS} ]; then
    cd native_client
    CFLAGS="-L${DS_TFDIR}/bazel-bin/tensorflow -L${DS_TFDIR}/bazel-bin/native_client" python ./setup.py bdist_wheel
    cd ..
fi
