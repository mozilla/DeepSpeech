#!/bin/bash

set -xe

source $(dirname "$0")/tc-tests-utils.sh

mkdir -p ${TASKCLUSTER_ARTIFACTS} || true

tar -C ${DS_ROOT_TASK}/DeepSpeech/tf/bazel-bin/tensorflow/ \
	-cf ${TASKCLUSTER_ARTIFACTS}/native_client.tar \
	libtensorflow_cc.so

tar -C ${DS_ROOT_TASK}/DeepSpeech/tf/bazel-bin/native_client/ \
	-uf ${TASKCLUSTER_ARTIFACTS}/native_client.tar \
	libdeepspeech.so \
	libdeepspeech_utils.so \
	libctc_decoder_with_kenlm.so \
	generate_trie

tar -C ${DS_ROOT_TASK}/DeepSpeech/ds/native_client/ \
	-uf ${TASKCLUSTER_ARTIFACTS}/native_client.tar \
	deepspeech

if [ -d ${DS_ROOT_TASK}/DeepSpeech/ds/wheels ]; then
  cp ${DS_ROOT_TASK}/DeepSpeech/ds/wheels/* ${TASKCLUSTER_ARTIFACTS}/
fi

find ${DS_ROOT_TASK}/DeepSpeech/ds/native_client/javascript/ -type f -name "deepspeech-*.tgz" -exec cp {} ${TASKCLUSTER_ARTIFACTS}/ \;

pixz -9 ${TASKCLUSTER_ARTIFACTS}/native_client.tar ${TASKCLUSTER_ARTIFACTS}/native_client.tar.xz
rm ${TASKCLUSTER_ARTIFACTS}/native_client.tar

cp ${DS_ROOT_TASK}/DeepSpeech/ds/data/alphabet.txt ${TASKCLUSTER_ARTIFACTS}/
