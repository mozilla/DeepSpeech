#!/bin/bash

set -xe

source $(dirname "$0")/tc-tests-utils.sh

node --version

npm --version

npm install -g node-gyp@6.x

devDir=$DS_ROOT_TASK/node-gyp-cache/

mkdir -p ${devDir}

node-gyp list ${devDir}

for node in ${SUPPORTED_NODEJS_BUILD_VERSIONS}; do
    node-gyp install --devdir ${devDir} \
        --target=$node
    mkdir ${devDir}/${node}/x64/ || true
    curl -sSL https://nodejs.org/dist/v${node}/win-x64/node.lib -o ${devDir}/${node}/x64/node.lib
done;

for electron in ${SUPPORTED_ELECTRONJS_VERSIONS}; do
    node-gyp install --devdir  ${devDir} \
        --target=$electron \
        --disturl=https://electronjs.org/headers \
        --runtime=electron
    mkdir ${devDir}/${electron}/x64/ || true
    curl -sSL https://electronjs.org/headers/v${electron}/win-x64/node.lib -o ${devDir}/${electron}/x64/node.lib
done;
