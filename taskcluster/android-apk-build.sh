#!/bin/bash

set -xe

arm_flavor=$1

source $(dirname "$0")/tc-tests-utils.sh

source ${DS_ROOT_TASK}/DeepSpeech/tf/tc-vars.sh

android_install_sdk_platform

do_deepspeech_java_apk_build
