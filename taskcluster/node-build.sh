#!/bin/bash

set -xe

package_option=$1

source $(dirname "$0")/tc-tests-utils.sh

source $(dirname "$0")/tf_tc-vars.sh

do_deepspeech_npm_package "${package_option}"
