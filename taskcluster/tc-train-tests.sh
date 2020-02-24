#!/bin/bash

set -xe

source $(dirname "$0")/tc-tests-utils.sh

extract_python_versions "$1" "pyver" "pyver_pkg" "py_unicode_type" "pyconf" "pyalias"

bitrate=$2

mkdir -p ${TASKCLUSTER_ARTIFACTS} || true
mkdir -p /tmp/train || true
mkdir -p /tmp/train_tflite || true

virtualenv_activate "${pyalias}" "deepspeech"

set -o pipefail
pip install --upgrade pip==19.3.1 setuptools==45.0.0 wheel==0.33.6 | cat
pip install --upgrade -r ${HOME}/DeepSpeech/ds/requirements.txt | cat
set +o pipefail

pushd ${HOME}/DeepSpeech/ds/
    verify_ctcdecoder_url
popd

decoder_pkg_url=$(get_python_pkg_url ${pyver_pkg} ${py_unicode_type} "ds_ctcdecoder" "${DECODER_ARTIFACTS_ROOT}")
LD_LIBRARY_PATH=${PY37_LDPATH}:$LD_LIBRARY_PATH pip install --verbose --only-binary :all: ${PY37_SOURCE_PACKAGE} ${decoder_pkg_url} | cat

# Prepare correct arguments for training
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

# Easier to rename to that we can exercize the LDC93S1 importer code to
# generate the CSV file.
echo "Moving ${sample_name} to LDC93S1.wav"
mv "${DS_ROOT_TASK}/DeepSpeech/ds/data/smoke_test/${sample_name}" "${DS_ROOT_TASK}/DeepSpeech/ds/data/smoke_test/LDC93S1.wav"

pushd ${HOME}/DeepSpeech/ds/
    # Run twice to test preprocessed features
    time ./bin/run-tc-ldc93s1_new.sh 249 "${sample_rate}"
    time ./bin/run-tc-ldc93s1_new.sh 1 "${sample_rate}"
    time ./bin/run-tc-ldc93s1_tflite.sh "${sample_rate}"
popd

cp /tmp/train/output_graph.pb ${TASKCLUSTER_ARTIFACTS}
cp /tmp/train_tflite/output_graph.tflite ${TASKCLUSTER_ARTIFACTS}

pushd ${HOME}/DeepSpeech/ds/
    python util/taskcluster.py --source tensorflow --artifact convert_graphdef_memmapped_format --branch r1.15 --target /tmp/
popd

/tmp/convert_graphdef_memmapped_format --in_graph=/tmp/train/output_graph.pb --out_graph=/tmp/train/output_graph.pbmm
cp /tmp/train/output_graph.pbmm ${TASKCLUSTER_ARTIFACTS}

pushd ${HOME}/DeepSpeech/ds/
    time ./bin/run-tc-ldc93s1_checkpoint.sh
popd

virtualenv_deactivate "${pyalias}" "deepspeech"
