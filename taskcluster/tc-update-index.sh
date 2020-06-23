#!/bin/bash
# Helper script because it is way too painful to deal with Windows' CMD.exe
# ways of escaping things when pushing JSON

set -xe

TC_EXPIRE=$1
TC_INSTANCE=$2
TC_INDEX=$3

source $(dirname "$0")/tc-tests-utils.sh

if [ ! -z "${TC_EXPIRE}" -a ! -z "${TC_INSTANCE}" -a ! -z "${TC_INDEX}" ]; then
    curl -sSL --fail -X PUT \
        -H "Content-Type: application/json" \
        -d "{\"taskId\":\"$TASK_ID\",\"rank\":0,\"expires\":\"${TC_EXPIRE}\",\"data\":{}}" \
        "http://${TC_INSTANCE}/index/v1/task/${TC_INDEX}"
fi;
