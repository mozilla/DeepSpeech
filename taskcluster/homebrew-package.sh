#!/bin/bash

set -xe

OS=$(uname)
if [ "${OS}" != "Darwin" ]; then
    echo "This should only run on OSX."
    exit 1
fi;

flavor=$1

source $(dirname "$0")/tc-tests-utils.sh

if [ -z "${TASKCLUSTER_TASK_DIR}" ]; then
  echo "No TASKCLUSTER_TASK_DIR, aborting."
  exit 1
fi

if [ "${flavor}" = "--builds" ]; then
  cd ${BUILDS_BREW}/ && tar -czf $TASKCLUSTER_ARTIFACTS/homebrew_builds.tar.gz .
fi;

if [ "${flavor}" = "--tests" ]; then
  cd ${TESTS_BREW}/ && tar -czf $TASKCLUSTER_ARTIFACTS/homebrew_tests.tar.gz .
fi;
