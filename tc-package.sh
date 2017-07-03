#!/bin/bash

set -xe

source $(dirname "$0")/tc-tests-utils.sh

mkdir -p ${TASKCLUSTER_ARTIFACTS} || true

target_tar="${TASKCLUSTER_ARTIFACTS}/native_client.tar"

ds_bin_dir="${DS_ROOT_TASK}/DeepSpeech/ds/native_client/"
nc_lib_dir="${DS_ROOT_TASK}/DeepSpeech/tf/bazel-bin/native_client/"
tf_lib_dir="${DS_ROOT_TASK}/DeepSpeech/tf/bazel-bin/tensorflow/"

# Create tar with deepspeech binary
tar -C ${ds_bin_dir} -cf "${target_tar}" deepspeech

# Populate with libdeepspeech* dependencies
for lib in libdeepspeech.so libdeepspeech_model.so libdeepspeech_utils.so;
do
    if [ -f "${nc_lib_dir}/${lib}" ]; then
        tar -C "${nc_lib_dir}" -uf "${target_tar}" ${lib}
    fi;
done;

# Populate with tensorflow dependencies
for lib in libtensorflow_cc.so compiler/aot/libruntime.so compiler/xla/service/cpu/libruntime_matmul.so compiler/xla/libexecutable_run_options.so;
do
    if [ -f "${tf_lib_dir}/${lib}" ]; then
        full_tf_dir="$(dirname ${tf_lib_dir}/${lib})"
        full_lib="$(basename ${tf_lib_dir}/${lib})"
        tar -C "${full_tf_dir}" -uf "${target_tar}" ${full_lib}
    fi;
done;

if [ -d ${DS_ROOT_TASK}/DeepSpeech/ds/wheels ]; then
    cp ${DS_ROOT_TASK}/DeepSpeech/ds/wheels/* ${TASKCLUSTER_ARTIFACTS}/
fi

find ${DS_ROOT_TASK}/DeepSpeech/ds/native_client/javascript/ -type f -name "deepspeech-*.tgz" -exec cp {} ${TASKCLUSTER_ARTIFACTS}/ \;

cp ${DS_ROOT_TASK}/DeepSpeech/ds/data/alphabet.txt ${TASKCLUSTER_ARTIFACTS}/

pixz -9 ${target_tar} ${TASKCLUSTER_ARTIFACTS}/native_client.tar.xz && rm ${target_tar}
