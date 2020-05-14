#!/bin/bash

set -x

tf_git_version=$(grep "STABLE_TF_GIT_VERSION" "bazel-out/stable-status.txt" | cut -d' ' -f2)
ds_version=$(grep "STABLE_DS_VERSION" "bazel-out/stable-status.txt" | cut -d' ' -f2)
ds_git_version=$(grep "STABLE_DS_GIT_VERSION" "bazel-out/stable-status.txt" | cut -d' ' -f2)
ds_graph_version=$(grep "STABLE_DS_GRAPH_VERSION" "bazel-out/stable-status.txt" | cut -d' ' -f2)

cat <<EOF
const char *tf_local_git_version() {
    return "${tf_git_version}";
}
const char *ds_version() {
    return "${ds_version}";
}
const char *ds_git_version() {
    return "${ds_git_version}";
}
const int ds_graph_version() {
    return ${ds_graph_version};
}
EOF
