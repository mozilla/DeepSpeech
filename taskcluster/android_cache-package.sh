#!/bin/bash

set -xe

TC_EXPIRE=$1
TC_INSTANCE=$2
TC_INDEX=$3

source $(dirname "$0")/tc-tests-utils.sh

cd $HOME/ && tar -czf $TASKCLUSTER_ARTIFACTS/android_cache.tar.gz DeepSpeech/Android/

if [ ! -z "${TC_EXPIRE}" -a ! -z "${TC_INSTANCE}" -a ! -z "${TC_INDEX}" ]; then
    curl -sSL --fail -X PUT \
        -H "Content-Type: application/json" \
        -d "{\"taskId\":\"$TASK_ID\",\"rank\":0,\"expires\":\"${TC_EXPIRE}\",\"data\":{}}" \
        "http://${TC_INSTANCE}/index/v1/task/${TC_INDEX}"
fi;
