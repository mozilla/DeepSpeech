#!/bin/bash

set -xe

source $(dirname "$0")/../tc-tests-utils.sh

source ${DS_ROOT_TASK}/DeepSpeech/tf/tc-vars.sh

do_deepspeech_npm_package
