#!/bin/bash

set -xe

mkdir -p /tmp/artifacts

tar -C ${HOME}/DeepSpeech/tf/bazel-bin/tensorflow/ \
	-cf /tmp/artifacts/native_client.tar \
	libtensorflow_cc.so

tar -C ${HOME}/DeepSpeech/tf/bazel-bin/native_client/ \
	-uf /tmp/artifacts/native_client.tar \
	libdeepspeech.so \
	libdeepspeech_utils.so

tar -C ${HOME}/DeepSpeech/ds/native_client/ \
	-uf /tmp/artifacts/native_client.tar \
	deepspeech

if [ -d ${HOME}/DeepSpeech/ds/native_client/dist ]; then
  cp ${HOME}/DeepSpeech/ds/native_client/dist/* /tmp/artifacts/
fi

find ${HOME}/DeepSpeech/ds/native_client/javascript/ -type f -name "deepspeech-*.tgz" -exec cp {} /tmp/artifacts/ \;

pixz -9 /tmp/artifacts/native_client.tar /tmp/artifacts/native_client.tar.xz
rm /tmp/artifacts/native_client.tar
