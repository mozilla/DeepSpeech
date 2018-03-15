#!/bin/sh

GIT_DIR="$(realpath "$(dirname "$(realpath "$0")")/../.git/")"
if [ ! -d "${GIT_DIR}" ]; then
   return 1
fi;

GIT_VERSION=$(git --git-dir="${GIT_DIR}" describe --long --tags)
if [ $? -ne 0 ]; then
   GIT_VERSION=unknown;
fi

cat <<EOF
#include <string>
const char* ds_git_version() {
  return "${GIT_VERSION}";
}
EOF
