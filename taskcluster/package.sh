#!/bin/bash

set -xe

source $(dirname "$0")/tc-tests-utils.sh

mkdir -p ${TASKCLUSTER_ARTIFACTS} || true

cp ${DS_DSDIR}/tensorflow/bazel*.log ${TASKCLUSTER_ARTIFACTS}/

package_native_client "native_client.tar.xz"

package_libdeepspeech_as_zip "libdeepspeech.zip"

if [ -d ${DS_DSDIR}/wheels ]; then
    cp ${DS_DSDIR}/wheels/* ${TASKCLUSTER_ARTIFACTS}/
    cp ${DS_DSDIR}/native_client/javascript/deepspeech-*.tgz ${TASKCLUSTER_ARTIFACTS}/
fi;

if [ -f ${DS_DSDIR}/native_client/javascript/wrapper.tar.gz ]; then
    cp ${DS_DSDIR}/native_client/javascript/wrapper.tar.gz ${TASKCLUSTER_ARTIFACTS}/
fi;
