#!/bin/bash

set -xe

arch=$1

source $(dirname "$0")/tc-tests-utils.sh

mkdir -p ${TASKCLUSTER_ARTIFACTS} || true

cp ${DS_ROOT_TASK}/DeepSpeech/ds/tensorflow/bazel*.log ${TASKCLUSTER_ARTIFACTS}/

package_native_client "native_client.tar.xz"

package_libdeepspeech_as_zip "libdeepspeech.zip"

case $arch in
"--x86_64")
  ${TAR} -cf - \
         -C ${DS_ROOT_TASK}/DeepSpeech/ds/native_client/swift/DerivedData/Build/Products/Release-iphonesimulator/deepspeech_ios.framework \
         | ${XZ} > "${TASKCLUSTER_ARTIFACTS}/ deepspeech_ios.framework.x86_64.tar.xz"
  ;;
"--arm64")
  ${TAR} -cf - \
         -C ${DS_ROOT_TASK}/DeepSpeech/ds/native_client/swift/DerivedData/Build/Products/Release-iphoneos/deepspeech_ios.framework \
         | ${XZ} > "${TASKCLUSTER_ARTIFACTS}/ deepspeech_ios.framework.arm64.tar.xz"
;;
esac
