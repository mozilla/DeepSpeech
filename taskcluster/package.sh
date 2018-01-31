#!/bin/bash

set -xe

source $(dirname "$0")/../tc-tests-utils.sh

mkdir -p ${TASKCLUSTER_ARTIFACTS} || true

cp ${DS_ROOT_TASK}/DeepSpeech/tf/bazel*.log ${TASKCLUSTER_ARTIFACTS}/

package_native_client "native_client.tar.xz"

if [ -d ${DS_ROOT_TASK}/DeepSpeech/ds/wheels ]; then
    cp ${DS_ROOT_TASK}/DeepSpeech/ds/wheels/* ${TASKCLUSTER_ARTIFACTS}/
    cp ${DS_ROOT_TASK}/DeepSpeech/ds/native_client/javascript/deepspeech-*.tgz ${TASKCLUSTER_ARTIFACTS}/
fi;

if [ -f ${DS_ROOT_TASK}/DeepSpeech/ds/native_client/javascript/wrapper.tar.gz ]; then
    cp ${DS_ROOT_TASK}/DeepSpeech/ds/native_client/javascript/wrapper.tar.gz ${TASKCLUSTER_ARTIFACTS}/
fi;
