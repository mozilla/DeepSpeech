#!/bin/bash

set -ex

curdir=$(dirname "$0")

pip3 install --quiet --user --upgrade pip

curl -L --silent https://raw.githubusercontent.com/lissyx/taskcluster-github-decision/0ee93e9f36ef1fc77218a96e865daa80d36b87e5/requirements.txt | pip3 install --quiet --user --upgrade -r /dev/stdin
curl -L --silent https://raw.githubusercontent.com/lissyx/taskcluster-github-decision/0ee93e9f36ef1fc77218a96e865daa80d36b87e5/tc-decision.py > ${curdir}/tc-decision.py

# First, perform dry run for push and pull request
# This should help us track merge failures in advance
for event in pull_request.opened pull_request.synchronize pull_request.reopened push;
do
    GITHUB_EVENT="${event}" TASK_ID="" GITHUB_HEAD_REF="refs/heads/branchName" python3 ${curdir}/tc-decision.py --dry
    GITHUB_EVENT="${event}" TASK_ID="" GITHUB_HEAD_REF="refs/tags/tagName" python3 ${curdir}/tc-decision.py --dry
done;

python3 ${curdir}/tc-decision.py
