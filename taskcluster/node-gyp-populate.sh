#!/bin/bash

set -xe

source $(dirname "$0")/tc-tests-utils.sh

node --version

npm --version

npm install node-gyp@6.x

export PATH=$HOME/node_modules/.bin/:$PATH

devDir=$DS_ROOT_TASK/node-gyp-cache/
nodejsDevDir=${devDir}/nodejs/
electronjsDevDir=${devDir}/electronjs/

mkdir -p ${nodejsDevDir} ${electronjsDevDir}
node-gyp list ${nodejsDevDir} ${electronjsDevDir}

for node in ${SUPPORTED_NODEJS_BUILD_VERSIONS}; do
    node-gyp install --devdir ${nodejsDevDir} \
        --target=$node
    mkdir ${nodejsDevDir}/${node}/x64/ || true
    curl -sSL https://nodejs.org/dist/v${node}/win-x64/node.lib -o ${nodejsDevDir}/${node}/x64/node.lib
done;

mkdir -p ${electronjsDevDir}
node-gyp list ${electronjsDevDir}

for electron in ${SUPPORTED_ELECTRONJS_VERSIONS}; do
    node-gyp install --devdir  ${electronjsDevDir} \
        --target=$electron \
        --disturl=https://electronjs.org/headers \
        --runtime=electron
    mkdir ${electronjsDevDir}/${electron}/x64/ || true
    curl -sSL https://electronjs.org/headers/v${electron}/win-x64/node.lib -o ${electronjsDevDir}/${electron}/x64/node.lib
done;
