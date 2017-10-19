#!/bin/bash

set -xe

source $(dirname "$0")/../tc-tests-utils.sh

mkdir -p ${TASKCLUSTER_ARTIFACTS} || true

tar -cf - \
    -C ${DS_TFDIR}/bazel-bin/native_client/ libctc_decoder_with_kenlm.so | pixz -9 > "${TASKCLUSTER_ARTIFACTS}/decoder.tar.xz"
