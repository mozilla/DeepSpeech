#!/bin/bash

set -xe

source $(dirname "$0")/tc-tests-utils.sh

mkdir -p ${TASKCLUSTER_ARTIFACTS} || true

cp ${DS_DSDIR}/doc/html.zip ${TASKCLUSTER_ARTIFACTS}/doc-html.zip
