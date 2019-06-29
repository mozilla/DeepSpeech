#!/bin/bash
set -ex

# This script will be run bazel when building process starts to
# generate key-value information that represents the status of the
# workspace. The output should be like
#
# KEY1 VALUE1
# KEY2 VALUE2
#
# Keys starting with STABLE_ cause dependent rules to be re-run when their value
# changes.
#
# If the script exits with non-zero code, it's considered as a failure
# and the output will be discarded.

# The code below presents an implementation that works for git repository
tf_git_rev=$(git describe --long --tags)
echo "STABLE_TF_GIT_VERSION ${tf_git_rev}"

pushd native_client
ds_git_rev=$(git describe --long --tags)
echo "STABLE_DS_GIT_VERSION ${ds_git_rev}"
ds_graph_version=$(cat ../GRAPH_VERSION)
echo "STABLE_DS_GRAPH_VERSION ${ds_graph_version}"
popd
