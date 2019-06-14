#!/bin/bash

set -xe

source $(dirname "$0")/tc-tests-utils.sh

mkdir -p ${TASKCLUSTER_ARTIFACTS} || true

if [ -d ${DS_ROOT_TASK}/DeepSpeech/ds/wheels ]; then
    cp ${DS_ROOT_TASK}/DeepSpeech/ds/wheels/* ${TASKCLUSTER_ARTIFACTS}/
fi;

