#!/bin/bash

set -ex

curdir=$(dirname "$0")

pip3 install --quiet --user --upgrade pip

curl -L --silent https://raw.githubusercontent.com/lissyx/taskcluster-github-decision/${TC_DECISION_SHA}/requirements.txt | pip3 install --quiet --user --upgrade -r /dev/stdin
curl -L --silent https://raw.githubusercontent.com/lissyx/taskcluster-github-decision/${TC_DECISION_SHA}/tc-decision.py > ${curdir}/tc-decision.py

# First, perform dry run for push and pull request
# This should help us track merge failures in advance
for event in pull_request.opened pull_request.synchronize pull_request.reopened push;
do
    GITHUB_EVENT="${event}" TASK_ID="" GITHUB_HEAD_REF="refs/heads/branchName" python3 ${curdir}/tc-decision.py --dry
    GITHUB_EVENT="${event}" TASK_ID="" GITHUB_HEAD_REF="refs/tags/tagName" python3 ${curdir}/tc-decision.py --dry
done;

python3 ${curdir}/tc-decision.py
