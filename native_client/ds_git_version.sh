#!/bin/sh

if [ `uname` = "Darwin" ]; then
   export PATH="/Users/build-user/TaskCluster/Workdir/tasks/tc-workdir/homebrew/opt/coreutils/libexec/gnubin:${PATH}"
fi

if [ `uname -o` = "Msys" ]; then
   export PATH="/c/Program Files/Git/bin/:${PATH}"
fi

DS_GIT_DIR="$(realpath "$(dirname "$(realpath "$0")")/../.git/")"
if [ ! -d "${DS_GIT_DIR}" ]; then
   return 1
fi;

TF_GIT_DIR="$(realpath $(pwd)/tensorflow/../.git/)"
if [ ! -d "${TF_GIT_DIR}" ]; then
   return 1
fi;

DS_GIT_VERSION=$(git --git-dir="${DS_GIT_DIR}" describe --long --tags)
if [ $? -ne 0 ]; then
   DS_GIT_VERSION=unknown;
fi

TF_GIT_VERSION=$(git --git-dir="${TF_GIT_DIR}" describe --long --tags)
if [ $? -ne 0 ]; then
   TF_GIT_VERSION=unknown;
fi

cat <<EOF
#include <string>
const char* ds_git_version() {
  return "${DS_GIT_VERSION}";
}
const char* tf_local_git_version() {
  return "${TF_GIT_VERSION}";
}
EOF
