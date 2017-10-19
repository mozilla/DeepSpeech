#!/bin/bash

set -xe

source $(dirname "$0")/../tc-tests-utils.sh

mkdir -p ${TASKCLUSTER_ARTIFACTS} || true

package_native_client "aot_model.tar.xz"
