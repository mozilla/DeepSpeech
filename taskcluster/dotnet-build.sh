#!/bin/bash

set -xe

package_option=$1

source $(dirname "$0")/tc-tests-utils.sh

source $(dirname "$0")/tf_tc-vars.sh


if [ "${package_option}" = "--cuda" ]; then
    PROJECT_NAME="DeepSpeech-GPU"
elif [ "${package_option}" = "--tflite" ]; then
    PROJECT_NAME="DeepSpeech-TFLite"
else
    PROJECT_NAME="DeepSpeech"
fi

do_nuget_repackage "${PROJECT_NAME}"
