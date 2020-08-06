#!/bin/bash

set -xe

arm_flavor=$1

source $(dirname "$0")/tc-tests-utils.sh

mkdir -p ${TASKCLUSTER_ARTIFACTS} || true

cp ${DS_ROOT_TASK}/DeepSpeech/ds/tensorflow/bazel*.log ${TASKCLUSTER_ARTIFACTS}/

cp ${DS_ROOT_TASK}/DeepSpeech/ds/native_client/java/app/build/outputs/apk/release/app*.apk ${TASKCLUSTER_ARTIFACTS}/
cp ${DS_ROOT_TASK}/DeepSpeech/ds/native_client/java/libmozillavoicestt/build/outputs/aar/libmozillavoicestt*.aar ${TASKCLUSTER_ARTIFACTS}/
cp ${DS_ROOT_TASK}/DeepSpeech/ds/native_client/java/libmozillavoicestt/build/libmozillavoicestt-*.maven.zip ${TASKCLUSTER_ARTIFACTS}/
