#!/bin/bash

set -xe

source $(dirname "$0")/tc-tests-utils.sh

kenlm_url=$1

maybe_py_or_android="$2"
maybe_android=""
maybe_py=""
if [ ! -z "${maybe_py_or_android}" -a "${maybe_py_or_android}" != "android" ]; then
    maybe_py=${maybe_py_or_android}
    extract_python_versions "${maybe_py}" "pyver" "pyver_pkg" "py_unicode_type" "pyconf" "pyalias"
fi

if [ "${maybe_py_or_android}" = "android" ]; then
    maybe_android="y"
    arm_flavor=$3
    api_level=$4
fi;

mkdir -p ${TASKCLUSTER_ARTIFACTS} || true
mkdir -p ${TASKCLUSTER_TMP_DIR}/scorer/bins ${TASKCLUSTER_TMP_DIR}/scorer/data || true

generic_download_targz ${TASKCLUSTER_TMP_DIR}/scorer/bins/ "${kenlm_url}"
download_native_client_files ${TASKCLUSTER_TMP_DIR}/scorer/bins/

export PATH=${TASKCLUSTER_TMP_DIR}/scorer/bins/:$PATH

if [ ! -z "${pyalias}" ]; then
    maybe_setup_virtualenv_cross_arm "${pyalias}" "deepspeech"
    virtualenv_activate "${pyalias}" "deepspeech"
fi

if [ "${maybe_android}" = "y" ]; then
    android_start_emulator "${arm_flavor}" "${api_level}"
fi

pushd ${DS_DSDIR}
    SCORER_DATA_DIR=${TASKCLUSTER_TMP_DIR}/scorer/data

    cp data/smoke_test/vocab.txt ${SCORER_DATA_DIR}
    cp data/smoke_test/vocab.txt ${SCORER_DATA_DIR}/vocab-500.txt
    cp data/alphabet.txt ${SCORER_DATA_DIR}
    gzip ${SCORER_DATA_DIR}/vocab.txt

    if [ "${maybe_android}" = "y" ]; then
        adb shell mkdir ${ANDROID_TMP_DIR}/ds/ ${ANDROID_TMP_DIR}/scorer/
        adb push ${SCORER_DATA_DIR}/ ${ANDROID_TMP_DIR}/scorer/
        adb push ${TASKCLUSTER_TMP_DIR}/scorer/bins/* ${ANDROID_TMP_DIR}/ds/

	SCORER_DATA_DIR=${ANDROID_TMP_DIR}/scorer/data
    fi

    if [ ! -z "${maybe_py}" ]; then
        pip install --only-binary :all: progressbar2

        python data/lm/generate_lm.py \
            --input_txt ${SCORER_DATA_DIR}/vocab.txt.gz \
            --output_dir ${SCORER_DATA_DIR}/ \
            --top_k 500 \
            --kenlm_bins ${TASKCLUSTER_TMP_DIR}/scorer/bins/ \
            --arpa_order 5 \
            --max_arpa_memory "85%" \
            --arpa_prune "0|0|1" \
            --binary_a_bits 255 \
            --binary_q_bits 8 \
            --binary_type trie

        ls -hal ${SCORER_DATA_DIR}
    fi

    if [ "${maybe_android}" = "y" ]; then
        ${DS_BINARY_PREFIX}lmplz \
            --memory 64M \
            --order 2 \
            --discount_fallback \
            --text ${SCORER_DATA_DIR}/vocab-500.txt \
            --arpa ${SCORER_DATA_DIR}/lm.arpa

        ${DS_BINARY_PREFIX}build_binary \
            -a 255 -q 8 -v trie \
            ${SCORER_DATA_DIR}/lm.arpa \
            ${SCORER_DATA_DIR}/lm.binary
    fi

    ${DS_BINARY_PREFIX}generate_scorer_package \
        --alphabet ${SCORER_DATA_DIR}/alphabet.txt \
        --lm ${SCORER_DATA_DIR}/lm.binary \
        --vocab ${SCORER_DATA_DIR}/vocab-500.txt \
        --package ${SCORER_DATA_DIR}/kenlm.scorer \
        --default_alpha 0.5 \
        --default_beta 1.25

    if [ "${maybe_android}" = "y" ]; then
        adb pull ${SCORER_DATA_DIR}/kenlm.scorer ${TASKCLUSTER_TMP_DIR}/scorer/data/
    fi

    ls -hal ${TASKCLUSTER_TMP_DIR}/scorer/data/kenlm.scorer
popd

if [ ! -z "${pyalias}" ]; then
    virtualenv_deactivate "${pyalias}" "deepspeech"
fi

if [ "${maybe_android}" = "y" ]; then
    android_stop_emulator
fi
