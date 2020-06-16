#!/bin/bash

set -xe

source $(dirname "$0")/tc-tests-utils.sh

cd ${GRADLE_USER_HOME}/../ && tar -czf $TASKCLUSTER_ARTIFACTS/gradle.tar.gz gradle-cache/
