#!/bin/bash

set -xe

source $(dirname "$0")/../tc-tests-utils.sh

mkdir -p ${TASKCLUSTER_ARTIFACTS} || true

package_native_client "native_client.tar.xz"

if [ -d ${DS_ROOT_TASK}/DeepSpeech/ds/python_dist_files ]; then
    # Python dist files
    cp ${DS_ROOT_TASK}/DeepSpeech/ds/python_dist_files/* ${TASKCLUSTER_ARTIFACTS}/

    # NodeJS package
    cp ${DS_ROOT_TASK}/DeepSpeech/ds/native_client/javascript/deepspeech-*.tgz ${TASKCLUSTER_ARTIFACTS}/
fi;
