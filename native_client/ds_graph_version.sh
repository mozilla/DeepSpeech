#!/bin/bash

if [ `uname` = "Darwin" ]; then
   export PATH="/Users/build-user/TaskCluster/Workdir/tasks/tc-workdir/homebrew/opt/coreutils/libexec/gnubin:${PATH}"
fi

DS_DIR="$(realpath "$(dirname "$(realpath "$0")")/../")"
if [ ! -d "${DS_DIR}" ]; then
   exit 1
fi;

DS_GRAPH_VERSION=$(cat "${DS_DIR}/GRAPH_VERSION")
if [ $? -ne 0 ]; then
   exit 1
fi

cat <<EOF
#define DS_GRAPH_VERSION ${DS_GRAPH_VERSION}
EOF
