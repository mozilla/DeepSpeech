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

bitrate=$3
set_ldc_sample_filename "${bitrate}"

model_source=${DEEPSPEECH_TEST_MODEL//.pb/.tflite}
model_name=$(basename "${model_source}")
model_name_mmap=$(basename "${model_source}")

download_data

node --version
npm --version

NODE_ROOT="${DS_ROOT_TASK}/ds-test/"
NODE_CACHE="${DS_ROOT_TASK}/ds-test.cache/"
export NODE_PATH="${NODE_ROOT}/node_modules/"
export PATH="${NODE_ROOT}:${NODE_PATH}/.bin/:${NODE_PATH}/electron/dist/:$PATH"

# make sure that NODE_ROOT really exists
mkdir -p ${NODE_ROOT}

npm install --prefix ${NODE_ROOT} --cache ${NODE_CACHE} electron@${electronver}

deepspeech_npm_url=$(get_dep_npm_pkg_url)
npm install --prefix ${NODE_ROOT} --cache ${NODE_CACHE} ${deepspeech_npm_url}

if [ "${OS}" = "Darwin" ]; then
  ln -s Electron.app/Contents/MacOS/Electron "${NODE_ROOT}/node_modules/electron/dist/node"
else
  ln -s electron "${NODE_ROOT}/node_modules/electron/dist/node"
  if [ -f "${NODE_ROOT}/node_modules/electron/dist//chrome-sandbox" ]; then
    export ELECTRON_DISABLE_SANDBOX=1
  fi;
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
