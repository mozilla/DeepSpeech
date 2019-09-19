#!/bin/bash

set -xe

THIS=$(dirname "$0")

source ../../taskcluster/tc-tests-utils.sh

DEP_TASK_ID=$(curl -s https://queue.taskcluster.net/v1/task/${TASK_ID} | python -c 'import json; import sys; print(" ".join(json.loads(sys.stdin.read())["dependencies"]));')

get_python_wheel_url()
{
  local this_python_version=$1

  extract_python_versions "${this_python_version}" "pyver" "pyver_pkg" "py_unicode_type" "pyconf" "pyalias"

  echo "$(get_python_pkg_url "${pyver_pkg}" "${py_unicode_type}" "deepspeech" https://queue.taskcluster.net/v1/task/${DEP_TASK_ID}/artifacts/public)"
}

get_npm_package_url()
{
  echo "https://queue.taskcluster.net/v1/task/${DEP_TASK_ID}/artifacts/public/deepspeech-${DS_VERSION}.tgz"
}
