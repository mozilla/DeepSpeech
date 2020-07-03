#!/bin/bash

set -ex

curdir=$(dirname "$0")/

pip3 install --quiet --user --upgrade pip

export PATH=$HOME/.local/bin/:$PATH

pip3 install --quiet --user --upgrade -r ${curdir}/tc-decision_reqs.txt

# First, perform dry run for push and pull request
# This should help us track merge failures in advance
for event in pull_request.opened pull_request.synchronize pull_request.reopened push;
do
    GITHUB_EVENT="${event}" TASK_ID="aa" GITHUB_HEAD_BRANCHORTAG="branchName" GITHUB_HEAD_REF="refs/heads/branchName" python3 ${curdir}/tc-decision.py --dry
done;

GITHUB_EVENT="tag" TASK_ID="aa" GITHUB_HEAD_BRANCHORTAG="tagName" GITHUB_HEAD_REF="refs/tags/tagName" python3 ${curdir}/tc-decision.py --dry

# Create a new env variable for usage in TaskCluster .yml files
export GITHUB_HEAD_BRANCHORTAG="${GITHUB_HEAD_BRANCH}${GITHUB_HEAD_TAG}"

# Quick hack because tc-decision uses GITHUB_HEAD_BRANCH
export GITHUB_HEAD_BRANCH="${GITHUB_HEAD_BRANCH}${GITHUB_HEAD_TAG}"
python3 ${curdir}/tc-decision.py
