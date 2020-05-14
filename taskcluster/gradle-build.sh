#!/bin/bash

set -xe

source $(dirname "$0")/tc-tests-utils.sh

if [ -z "${GRADLE_USER_HOME}" ]; then
  echo "Unable to generate cache without an emplacement"
  exit 1
fi;

mkdir -p ${GRADLE_USER_HOME}

export ANDROID_HOME=${ANDROID_SDK_HOME}

# Gradle likes to play with us.
android_install_ndk
android_install_sdk

pushd ${DS_ROOT_TASK}/DeepSpeech/ds/native_client/java/
  ./gradlew androidDependencies dependentComponents dependencies
  # we need this for aapt2 binary
  ./gradlew buildNeeded || true # will try javac which is doomed to fail
  ./gradlew --refresh-dependencies
popd

du -hs ${GRADLE_USER_HOME}/*
