#!/bin/bash

set -xe

source $(dirname "$0")/tc-tests-utils.sh

nodever=$1
electronver=$2

if [ -z "${nodever}" ]; then
    echo "No node version given, aborting."
    exit 1
fi;

if [ -z "${electronver}" ]; then
    echo "No electron version given, aborting."
    exit 1
fi;

download_data

node --version
npm --version

NODE_ROOT="${DS_ROOT_TASK}/ds-test/"
export NODE_PATH="${NODE_ROOT}/node_modules/"
export PATH="${NODE_ROOT}:${NODE_PATH}/.bin/:${NODE_PATH}/electron/dist/:$PATH"

npm install --prefix ${NODE_ROOT} electron@${electronver}

npm install --prefix ${NODE_ROOT} ${DEEPSPEECH_NODEJS}/deepspeech-${DS_VERSION}.tgz

if [ "${OS}" = "Darwin" ]; then
  ln -s Electron.app/Contents/MacOS/Electron "${NODE_ROOT}/node_modules/electron/dist/node"
else
  ln -s electron "${NODE_ROOT}/node_modules/electron/dist/node"
fi

find ${NODE_ROOT}/node_modules/electron/dist/

which electron
which node

if [ "${OS}" = "Linux" ]; then
  export DISPLAY=':99.0'
  Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &
  xvfb_process=$!
fi

node --version

check_runtime_electronjs

run_electronjs_inference_tests

if [ "${OS}" = "Linux" ]; then
  sleep 1
  kill -9 ${xvfb_process} || true
fi
