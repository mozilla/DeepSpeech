#!/bin/bash

set -xe

source $(dirname "$0")/tc-tests-utils.sh

extract_python_versions "$1" "pyver" "pyver_pkg" "py_unicode_type" "pyconf" "pyalias"

bitrate=$2

mkdir -p ${TASKCLUSTER_ARTIFACTS} || true
mkdir -p ${TASKCLUSTER_TMP_DIR}/ckpt || true

download_dependency_file "checkpoint.tar.xz"
cd ${TASKCLUSTER_TMP_DIR} && ${UNXZ} checkpoint.tar.xz
cd ${TASKCLUSTER_TMP_DIR}/ckpt/ && tar -xf ${TASKCLUSTER_TMP_DIR}/checkpoint.tar

virtualenv_activate "${pyalias}" "deepspeech"

set -o pipefail
pip install --upgrade pip==19.3.1 setuptools==45.0.0 wheel==0.33.6 | cat
pushd ${HOME}/DeepSpeech/ds
    pip install --upgrade . | cat
    pip install --upgrade -r requirements_transcribe.txt | cat
popd
set +o pipefail

# Prepare correct arguments for transcribeing
case "${bitrate}" in
    8k)
        sample_rate=8000
        sample_name='LDC93S1_pcms16le_1_8000.wav'
    ;;
    16k)
        sample_rate=16000
        sample_name='LDC93S1_pcms16le_1_16000.wav'
    ;;
esac

pushd ${HOME}/DeepSpeech/ds/
    python transcribe.py \
	    --src "${DS_ROOT_TASK}/DeepSpeech/ds/data/smoke_test/${sample_name}" \
	    --dst ${TASKCLUSTER_ARTIFACTS}/transcribe.log \
	    --n_hidden 100 \
	    --checkpoint_dir ${TASKCLUSTER_TMP_DIR}/ckpt/ \
	    --scorer "${DS_ROOT_TASK}/DeepSpeech/ds/data/smoke_test/pruned_lm.scorer"
popd

virtualenv_deactivate "${pyalias}" "deepspeech"
