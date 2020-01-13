#!/bin/bash

set -xe

source $(dirname "$0")/tc-tests-utils.sh

exec_benchmark()
{
    model_file="$1"
    run_postfix=$2

    mkdir -p /tmp/bench-ds/ || true
    mkdir -p /tmp/bench-ds-nolm/ || true

    csv=${TASKCLUSTER_ARTIFACTS}/benchmark-${run_postfix}.csv
    csv_nolm=${TASKCLUSTER_ARTIFACTS}/benchmark-nolm-${run_postfix}.csv
    png=${TASKCLUSTER_ARTIFACTS}/benchmark-${run_postfix}.png
    svg=${TASKCLUSTER_ARTIFACTS}/benchmark-${run_postfix}.svg

    python ${DS_ROOT_TASK}/DeepSpeech/ds/bin/benchmark_nc.py \
        --dir /tmp/bench-ds/ \
        --models ${model_file} \
        --wav /tmp/LDC93S1.wav \
        --lm_binary /tmp/lm.binary \
        --trie /tmp/trie \
        --csv ${csv}

    python ${DS_ROOT_TASK}/DeepSpeech/ds/bin/benchmark_nc.py \
        --dir /tmp/bench-ds-nolm/ \
        --models ${model_file} \
        --wav /tmp/LDC93S1.wav \
        --csv ${csv_nolm}

    python ${DS_ROOT_TASK}/DeepSpeech/ds/bin/benchmark_plotter.py \
        --dataset "TaskCluster model" ${csv} \
        --dataset "TaskCluster model (no LM)" ${csv_nolm} \
        --title "TaskCluster model benchmark" \
        --wav /tmp/LDC93S1.wav \
        --plot ${png} \
        --size 1280x720

    python ${DS_ROOT_TASK}/DeepSpeech/ds/bin/benchmark_plotter.py \
        --dataset "TaskCluster model" ${csv} \
        --dataset "TaskCluster model (no LM)" ${csv_nolm} \
        --title "TaskCluster model benchmark" \
        --wav /tmp/LDC93S1.wav \
        --plot ${svg} \
        --size 1280x720
}

pyver=3.5.6

unset PYTHON_BIN_PATH
unset PYTHONPATH
export PYENV_ROOT="${HOME}/ds-test/.pyenv"
export PATH="${PYENV_ROOT}/bin:$PATH"

mkdir -p ${TASKCLUSTER_ARTIFACTS} || true
mkdir -p ${PYENV_ROOT} || true

# We still need to get model, wav and alphabet
download_data

# Follow benchmark naming from parameters in bin/run-tc-ldc93s1.sh
# Okay, it's not really the real LSTM sizes, just a way to verify how things
# actually behave.
for size in 100 200 300 400 500 600 700 800 900;
do
    cp /tmp/${model_name} /tmp/test.frozen.e75.lstm${size}.ldc93s1.pb
done;

# Let's make it a ZIP file. We don't want the directory structure.
zip --junk-paths -r9 /tmp/test.frozen.e75.lstm100-900.ldc93s1.zip /tmp/test.frozen.e75.lstm*.ldc93s1.pb && rm /tmp/test.frozen.e75.lstm*.ldc93s1.pb

# And prepare for multiple files on the CLI
model_list=""
for size in 10 20 30 40 50 60 70 80 90;
do
    cp /tmp/${model_name} /tmp/test.frozen.e75.lstm${size}.ldc93s1.pb
    model_list="${model_list} /tmp/test.frozen.e75.lstm${size}.ldc93s1.pb"
done;

# Let's prepare another model for single-model codepath
mv /tmp/${model_name} /tmp/test.frozen.e75.lstm494.ldc93s1.pb

export TASKCLUSTER_SCHEME=${DEEPSPEECH_ARTIFACTS_ROOT}/native_client.tar.xz

install_pyenv "${PYENV_ROOT}"
install_pyenv_virtualenv "$(pyenv root)/plugins/pyenv-virtualenv"

PYENV_NAME=deepspeech-test
pyenv install ${pyver}
pyenv virtualenv ${pyver} ${PYENV_NAME}
source ${PYENV_ROOT}/versions/${pyver}/envs/${PYENV_NAME}/bin/activate

set -o pipefail
pip install --upgrade pip==19.3.1 setuptools==45.0.0 wheel==0.33.6 | cat
pip install -r ${DS_ROOT_TASK}/DeepSpeech/ds/requirements.txt | cat
set +o pipefail

exec_benchmark "/tmp/test.frozen.e75.lstm494.ldc93s1.pb" "single-model"
exec_benchmark "/tmp/test.frozen.e75.lstm100-900.ldc93s1.zip" "zipfile-model"
exec_benchmark "${model_list}" "multi-model"

deactivate
pyenv uninstall --force ${PYENV_NAME}
