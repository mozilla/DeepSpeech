#!/bin/bash

set -xe

source $(dirname "$0")/tc-tests-utils.sh

mkdir -p ${TASKCLUSTER_ARTIFACTS} || true

target_tar="${TASKCLUSTER_ARTIFACTS}/native_client.tar"

ds_src_dir="${DS_ROOT_TASK}/DeepSpeech/ds/"
ds_bin_dir="${DS_ROOT_TASK}/DeepSpeech/ds/native_client/"
ds_lm_dir="${DS_ROOT_TASK}/DeepSpeech/ds/native_client/kenlm/"
nc_lib_dir="${DS_ROOT_TASK}/DeepSpeech/tf/bazel-bin/native_client/"
tf_lib_dir="${DS_ROOT_TASK}/DeepSpeech/tf/bazel-bin/tensorflow/"

ds_binary="deepspeech"
ds_data="LICENSE"
ds_dependencies="libdeepspeech.so libdeepspeech_model.so libdeepspeech_utils.so"
ds_lm_dependencies="libctc_decoder_with_kenlm.so generate_trie"
ds_lm_data="README.mozilla"
tf_dependencies="libtensorflow_cc.so compiler/aot/libruntime.so compiler/xla/service/cpu/libruntime_matmul.so compiler/xla/libexecutable_run_options.so"

# Create tar with deepspeech binary
tar -C ${ds_bin_dir} -cf "${target_tar}" ${ds_binary}

# Copy data files in the tar
for fdata in ${ds_data};
do
    if [ -f "${ds_src_dir}/${fdata}" ]; then
        tar -C "${ds_src_dir}" -uf "${target_tar}" ${fdata}
    fi;
done;

# Copy KenLM data files in the tar
for lm_data in ${ds_lm_data};
do
    if [ -f "${ds_lm_dir}/${lm_data}" ]; then
        tar -C "${ds_lm_dir}" -uf "${target_tar}" ${lm_data}
    fi;
done;

# Populate with libdeepspeech* dependencies
for lib in ${ds_dependencies} ${ds_lm_dependencies};
do
    if [ -f "${nc_lib_dir}/${lib}" ]; then
        tar -C "${nc_lib_dir}" -uf "${target_tar}" ${lib}
    fi;
done;

# Populate with tensorflow dependencies
for lib in ${tf_dependencies};
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
