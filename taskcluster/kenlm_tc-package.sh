#!/bin/bash

set -xe

source $(dirname $0)/tc-all-vars.sh

mkdir -p ${TASKCLUSTER_ARTIFACTS} || true

cd ${DS_ROOT_TASK}/DeepSpeech/ds/kenlm/build/bin/ && \
    tar \
        -czf ${TASKCLUSTER_ARTIFACTS}/kenlm.tar.gz \
        build_binary${PLATFORM_EXE_SUFFIX} \
        filter${PLATFORM_EXE_SUFFIX} \
        lmplz${PLATFORM_EXE_SUFFIX}
