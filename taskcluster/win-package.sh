#!/bin/bash

set -xe

arm_flavor=$1

source $(dirname "$0")/../tc-tests-utils.sh

mkdir -p ${TASKCLUSTER_ARTIFACTS} || true

cp ${DS_ROOT_TASK}/DeepSpeech/tf/bazel*.log ${TASKCLUSTER_ARTIFACTS}/

package_native_client "native_client.tar.xz"

cp ${DS_ROOT_TASK}/DeepSpeech/ds/examples/net_framework/CSharpExamples/*.nupkg ${TASKCLUSTER_ARTIFACTS}/

cp ${DS_ROOT_TASK}/DeepSpeech/ds/examples/net_framework/CSharpExamples/DeepSpeechConsole/bin/x64/Release/DeepSpeechConsole.exe ${TASKCLUSTER_ARTIFACTS}/

if [ -f ${DS_ROOT_TASK}/DeepSpeech/ds/native_client/javascript/wrapper.tar.gz ]; then
    cp ${DS_ROOT_TASK}/DeepSpeech/ds/native_client/javascript/wrapper.tar.gz ${TASKCLUSTER_ARTIFACTS}/
    cp ${DS_ROOT_TASK}/DeepSpeech/ds/native_client/javascript/deepspeech-*.tgz ${TASKCLUSTER_ARTIFACTS}/
fi;
