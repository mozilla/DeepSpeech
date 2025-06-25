#!/bin/bash

set -xe

source "$(dirname "$0")/tf-vars.sh"

mkdir -p "${CI_ARTIFACTS_DIR}" || true

cp "${DS_ROOT_TASK}/tensorflow/bazel_*.log" "${CI_ARTIFACTS_DIR}" || true

OUTPUT_ROOT="${DS_ROOT_TASK}/tensorflow/bazel-bin"

for output_bin in \
    tensorflow/lite/experimental/c/libtensorflowlite_c.so \
    tensorflow/tools/graph_transforms/transform_graph \
    tensorflow/tools/graph_transforms/summarize_graph \
    tensorflow/tools/benchmark/benchmark_model \
    tensorflow/contrib/util/convert_graphdef_memmapped_format \
    tensorflow/lite/toco/toco;
do
    if [ -f "${OUTPUT_ROOT}/${output_bin}" ]; then
        cp "${OUTPUT_ROOT}/${output_bin}" "${CI_ARTIFACTS_DIR}/"
    fi
done

if [ -f "${OUTPUT_ROOT}/tensorflow/lite/tools/benchmark/benchmark_model" ]; then
    cp "${OUTPUT_ROOT}/tensorflow/lite/tools/benchmark/benchmark_model" "${CI_ARTIFACTS_DIR}/lite_benchmark_model"
fi

TAR_EXCLUDE="--exclude=./dls/*"
if [ "${OS}" = "Darwin" ]; then
    TAR_EXCLUDE="--exclude=./dls/* --exclude=./public/* --exclude=./generic-worker/* --exclude=./homebrew/* --exclude=./homebrew.cache/* --exclude=./homebrew.logs/*"
fi

if [ "${OS}" = "${CI_MSYS_VERSION}" ]; then
    export PATH=$PATH:'/c/Program Files/7-Zip/'
    7z a '-xr!.\dls\' '-xr!.\tmp\' '-xr!.\msys64\' -snl -snh -so home.tar . | 7z a -si "${CI_ARTIFACTS_DIR}/home.tar.xz"
else
    tar -C "${DS_ROOT_TASK}" ${TAR_EXCLUDE} -cf - . | xz > "${CI_ARTIFACTS_DIR}/home.tar.xz"
fi

if [ "${OS}" = "Linux" ]; then
    SHA_SUM_GEN="sha256sum"
elif [ "${OS}" = "${CI_MSYS_VERSION}" ]; then
    SHA_SUM_GEN="sha256sum"
elif [ "${OS}" = "Darwin" ]; then
    SHA_SUM_GEN="shasum -a 256"
fi

${SHA_SUM_GEN} "${CI_ARTIFACTS_DIR}"/* > "${CI_ARTIFACTS_DIR}/checksums.txt"

