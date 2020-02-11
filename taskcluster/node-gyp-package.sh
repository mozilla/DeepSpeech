#!/bin/bash

set -xe

source $(dirname "$0")/tc-tests-utils.sh

mkdir -p ${TASKCLUSTER_ARTIFACTS} || true

cd $DS_ROOT_TASK/node-gyp-cache/ && tar -czf ${TASKCLUSTER_ARTIFACTS}/node-gyp-cache.tar.gz .
