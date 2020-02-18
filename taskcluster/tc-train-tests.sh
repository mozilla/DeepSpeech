#!/bin/bash

set -xe

source $(dirname "$0")/tc-tests-utils.sh

pyver_full=$1
bitrate=$2

if [ -z "${pyver_full}" ]; then
    echo "No python version given, aborting."
    exit 1
fi;

pyver=$(echo "${pyver_full}" | cut -d':' -f1)

# 2.7.x => 27
pyver_pkg=$(echo "${pyver}" | cut -d'.' -f1,2 | tr -d '.')

py_unicode_type=$(echo "${pyver_full}" | cut -d':' -f2)
if [ "${py_unicode_type}" = "m" ]; then
  pyconf="ucs2"
elif [ "${py_unicode_type}" = "mu" ]; then
  pyconf="ucs4"
fi;

unset PYTHON_BIN_PATH
unset PYTHONPATH
export PYENV_ROOT="${HOME}/ds-train/.pyenv"
export PATH="${PYENV_ROOT}/bin:${HOME}/bin:$PATH"

mkdir -p ${PYENV_ROOT} || true
mkdir -p ${TASKCLUSTER_ARTIFACTS} || true
mkdir -p /tmp/train || true
mkdir -p /tmp/train_tflite || true

install_pyenv "${PYENV_ROOT}"
install_pyenv_virtualenv "$(pyenv root)/plugins/pyenv-virtualenv"

PYENV_NAME=deepspeech-train
PYTHON_CONFIGURE_OPTS="--enable-unicode=${pyconf}" pyenv install ${pyver}
pyenv virtualenv ${pyver} ${PYENV_NAME}
source ${PYENV_ROOT}/versions/${pyver}/envs/${PYENV_NAME}/bin/activate

set -o pipefail
pip install --upgrade pip==19.3.1 setuptools==45.0.0 wheel==0.33.6 | cat
pip install --upgrade -r ${HOME}/DeepSpeech/ds/requirements.txt | cat
set +o pipefail

pushd ${HOME}/DeepSpeech/ds/
    verify_ctcdecoder_url
popd

platform=$(python -c 'import sys; import platform; plat = platform.system().lower(); arch = platform.machine().lower(); plat = "manylinux1" if plat == "linux" and arch == "x86_64" else plat; plat = "macosx_10_10" if plat == "darwin" else plat; sys.stdout.write("%s_%s" % (plat, platform.machine()));')
whl_ds_version="$(python -c 'from pkg_resources import parse_version; print(parse_version("'${DS_VERSION}'"))')"
decoder_pkg="ds_ctcdecoder-${whl_ds_version}-cp${pyver_pkg}-cp${pyver_pkg}${py_unicode_type}-${platform}.whl"

decoder_pkg_url=${DECODER_ARTIFACTS_ROOT}/${decoder_pkg}

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

deactivate
