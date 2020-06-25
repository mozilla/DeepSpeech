#!/bin/bash

set -xe

arm_flavor=$1

source $(dirname "$0")/tc-tests-utils.sh

mkdir -p ${TASKCLUSTER_ARTIFACTS} || true

cp ${DS_ROOT_TASK}/DeepSpeech/ds/tensorflow/bazel*.log ${TASKCLUSTER_ARTIFACTS}/

package_native_client_ndk "native_client.tar.xz" "${arm_flavor}"
