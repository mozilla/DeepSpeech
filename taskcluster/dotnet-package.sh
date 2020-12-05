#!/bin/bash

set -xe

source $(dirname "$0")/tc-tests-utils.sh

mkdir -p ${TASKCLUSTER_ARTIFACTS} || true

# Dotnet Package
cp ${DS_ROOT_TASK}/DeepSpeech/ds/native_client/dotnet/*.nupkg ${TASKCLUSTER_ARTIFACTS}/
