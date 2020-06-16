#!/bin/bash

set -xe

source $(dirname "$0")/tc-tests-utils.sh

cd $HOME/ && tar -czf $TASKCLUSTER_ARTIFACTS/android_cache.tar.gz DeepSpeech/Android/
