#!/bin/bash

set -xe

source $(dirname "$0")/tc-tests-utils.sh

arm_flavor=$1
api_level=$2
api_kind=$3

export ANDROID_HOME=${ANDROID_SDK_HOME}

android_install_ndk

android_install_sdk

# Required for running APK tests later
android_install_sdk_platform "android-27"

if [ "${arm_flavor}" != "sdk" ]; then
  android_setup_emulator "${arm_flavor}" "${api_level}" "${api_kind}"
fi;
