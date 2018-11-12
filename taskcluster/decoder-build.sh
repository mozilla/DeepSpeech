#!/bin/bash

set -xe

source $(dirname "$0")/../tc-tests-utils.sh

export SYSTEM_TARGET=host

do_deepspeech_decoder_build
