#!/bin/bash

set -xe

source $(dirname "$0")/tc-tests-utils.sh

cd ${PYENV_ROOT}/ && $TAR -czf $TASKCLUSTER_ARTIFACTS/pyenv.tar.gz .
