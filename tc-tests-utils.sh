#!/bin/bash

set -xe

OS=$(uname)
if [ "${OS}" = "Linux" ]; then
    export DS_ROOT_TASK=${HOME}
fi;

if [ "${OS}" = "Darwin" ]; then
    export DS_ROOT_TASK=${TASKCLUSTER_TASK_DIR}
    export SWIG_LIB="$(find ${DS_ROOT_TASK}/homebrew/Cellar/swig/ -type f -name "swig.swg" | xargs dirname)"

    # It seems chaining |export DYLD_LIBRARY_PATH=...| does not work, maybe
    # because of SIP? Who knows ...
    if [ ! -z "${EXTRA_ENV}" ]; then
        eval "export ${EXTRA_ENV}"
    fi;
fi;

export TASKCLUSTER_ARTIFACTS=${TASKCLUSTER_ARTIFACTS:-/tmp/artifacts}
export TASKCLUSTER_TMP_DIR=${TASKCLUSTER_TMP_DIR:-/tmp}

mkdir -p ${TASKCLUSTER_TMP_DIR} || true

export DS_TFDIR=${DS_ROOT_TASK}/DeepSpeech/tf
export DS_DSDIR=${DS_ROOT_TASK}/DeepSpeech/ds

export BAZEL_CTC_TARGETS="//native_client:libctc_decoder_with_kenlm.so"

export EXTRA_AOT_CFLAGS=""
export EXTRA_AOT_LDFLAGS=""
export EXTRA_AOT_LIBS="-ldeepspeech_model"

export BAZEL_AOT_BUILD_FLAGS="--define=DS_NATIVE_MODEL=1 --define=DS_MODEL_TIMESTEPS=64"
export BAZEL_AOT_TARGETS="
//native_client:libdeepspeech_model.so
"

export DS_VERSION="$(cat ${DS_DSDIR}/VERSION)"

model_source="${DEEPSPEECH_TEST_MODEL}"
model_name="$(basename "${model_source}")"
model_name_mmap="$(basename -s ".pb" "${model_source}").pbmm"
model_source_mmap="$(dirname "${model_source}")/${model_name_mmap}"

SUPPORTED_PYTHON_VERSIONS=${SUPPORTED_PYTHON_VERSIONS:-2.7.14:ucs2 2.7.14:ucs4 3.4.8:ucs4 3.5.5:ucs4 3.6.4:ucs4 3.7.0:ucs4}
SUPPORTED_NODEJS_VERSIONS=${SUPPORTED_NODEJS_VERSIONS:-4.9.1 5.12.0 6.14.1 7.10.1 8.11.1 9.11.1 10.3.0}

# This verify exact inference result
assert_correct_inference()
{
  phrase=$1
  expected=$2

  if [ -z "${phrase}" -o -z "${expected}" ]; then
      echo "One or more empty strings:"
      echo "phrase: <${phrase}>"
      echo "expected: <${expected}>"
      return 1
  fi;

  if [ "${phrase}" = "${expected}" ]; then
      echo "Proper output has been produced:"
      echo "${phrase}"
      return 0
  else
      echo "!! Non matching output !!"
      echo "got: <${phrase}>"
      echo "xxd:"; echo "${phrase}" | xxd
      echo "-------------------"
      echo "expected: <${expected}>"
      echo "xxd:"; echo "${expected}" | xxd
      return 1
  fi;
}

# This verify that ${expected} is contained within ${phrase}
assert_working_inference()
{
  phrase=$1
  expected=$2

  if [ -z "${phrase}" -o -z "${expected}" ]; then
      echo "One or more empty strings:"
      echo "phrase: <${phrase}>"
      echo "expected: <${expected}>"
      return 1
  fi;

  case "${phrase}" in
      *${expected}*)
          echo "Proper output has been produced:"
          echo "${phrase}"
          return 0
      ;;

      *)
          echo "!! Non matching output !!"
          echo "got: <${phrase}>"
          echo "xxd:"; echo "${phrase}" | xxd
          echo "-------------------"
          echo "expected: <${expected}>"
          echo "xxd:"; echo "${expected}" | xxd
          return 1
      ;;
  esac
}

assert_shows_something()
{
  stderr=$1
  expected=$2

  if [ -z "${stderr}" -o -z "${expected}" ]; then
      echo "One or more empty strings:"
      echo "stderr: <${stderr}>"
      echo "expected: <${expected}>"
      return 1
  fi;

  case "${stderr}" in
      *${expected}*)
          echo "Proper output has been produced:"
          echo "${stderr}"
          return 0
      ;;

      *)
          echo "!! Non matching output !!"
          echo "got: <${stderr}>"
          echo "xxd:"; echo "${stderr}" | xxd
          echo "-------------------"
          echo "expected: <${expected}>"
          echo "xxd:"; echo "${expected}" | xxd
          return 1
      ;;
  esac
}

assert_correct_ldc93s1()
{
  assert_correct_inference "$1" "she had your dark suit in greasy wash water all year"
}

assert_correct_multi_ldc93s1()
{
  assert_shows_something "$1" "/LDC93S1.wav%she had your dark suit in greasy wash water all year%"
  assert_shows_something "$1" "/LDC93S1_pcms16le_2_44100.wav%she had your dark suit in greasy wash water all year%"
  ## 8k will output garbage anyway ...
  # assert_shows_something "$1" "/LDC93S1_pcms16le_1_8000.wav%she hayorasryrtl lyreasy asr watal w water all year%"
}

assert_correct_ldc93s1_prodmodel_v1()
{
  assert_correct_inference "$1" "she had yeduckso in greasy wash for all year"
}

assert_correct_ldc93s1_prodmodel_v2()
{
  assert_correct_inference "$1" "she had a ducsuot in greasy wathorerall year"
}

assert_working_ldc93s1_prodmodel()
{
  assert_working_inference "$1" "she had"
}

assert_correct_ldc93s1_somodel()
{
    somodel_nolm=$1
    somodel_withlm=$2

    # We want to be able to return non zero value from the function, while not
    # failing the whole execution
    set +e

    assert_correct_ldc93s1 "${somodel_nolm}"
    so_nolm=$?

    assert_correct_ldc93s1 "${somodel_withlm}"
    so_lm=$?

    set -e

    # We accept that with no LM there may be errors, but we do not accept that
    # for LM. For now.
    if [ ${so_lm} -eq 1 ] && [ ${so_nolm} -eq 1 -o ${so_nolm} -eq 0 ];
    then
        exit 1
    elif [ ${so_lm} -eq 0 ] && [ ${so_nolm} -eq 1 -o ${so_nolm} -eq 0 ];
    then
        exit 0
    else
        echo "Unexpected status"
        exit 2
    fi
}

assert_correct_warning_upsampling()
{
  assert_shows_something "$1" "erratic speech recognition"
}

assert_tensorflow_version()
{
  assert_shows_something "$1" "${EXPECTED_TENSORFLOW_VERSION}"
}

check_tensorflow_version()
{
  set +e
  ds_help=$(deepspeech 2>&1 1>/dev/null)
  set -e

  assert_tensorflow_version "${ds_help}"
}

run_all_inference_tests()
{
  phrase_pbmodel_nolm=$(deepspeech --model ${TASKCLUSTER_TMP_DIR}/${model_name} --alphabet ${TASKCLUSTER_TMP_DIR}/alphabet.txt --audio ${TASKCLUSTER_TMP_DIR}/LDC93S1.wav)
  assert_correct_ldc93s1 "${phrase_pbmodel_nolm}"

  phrase_pbmodel_nolm=$(deepspeech --model ${TASKCLUSTER_TMP_DIR}/${model_name_mmap} --alphabet ${TASKCLUSTER_TMP_DIR}/alphabet.txt --audio ${TASKCLUSTER_TMP_DIR}/LDC93S1.wav)
  assert_correct_ldc93s1 "${phrase_pbmodel_nolm}"

  phrase_pbmodel_withlm=$(deepspeech --model ${TASKCLUSTER_TMP_DIR}/${model_name_mmap} --alphabet ${TASKCLUSTER_TMP_DIR}/alphabet.txt --lm ${TASKCLUSTER_TMP_DIR}/lm.binary --trie ${TASKCLUSTER_TMP_DIR}/trie --audio ${TASKCLUSTER_TMP_DIR}/LDC93S1.wav)
  assert_correct_ldc93s1 "${phrase_pbmodel_withlm}"

  phrase_pbmodel_nolm_stereo_44k=$(deepspeech --model ${TASKCLUSTER_TMP_DIR}/${model_name_mmap} --alphabet ${TASKCLUSTER_TMP_DIR}/alphabet.txt --audio ${TASKCLUSTER_TMP_DIR}/LDC93S1_pcms16le_2_44100.wav)
  assert_correct_ldc93s1 "${phrase_pbmodel_nolm_stereo_44k}"

  phrase_pbmodel_withlm_stereo_44k=$(deepspeech --model ${TASKCLUSTER_TMP_DIR}/${model_name_mmap} --alphabet ${TASKCLUSTER_TMP_DIR}/alphabet.txt --lm ${TASKCLUSTER_TMP_DIR}/lm.binary --trie ${TASKCLUSTER_TMP_DIR}/trie --audio ${TASKCLUSTER_TMP_DIR}/LDC93S1_pcms16le_2_44100.wav)
  assert_correct_ldc93s1 "${phrase_pbmodel_withlm_stereo_44k}"

  phrase_pbmodel_nolm_mono_8k=$(deepspeech --model ${TASKCLUSTER_TMP_DIR}/${model_name_mmap} --alphabet ${TASKCLUSTER_TMP_DIR}/alphabet.txt --audio ${TASKCLUSTER_TMP_DIR}/LDC93S1_pcms16le_1_8000.wav 2>&1 1>/dev/null)
  assert_correct_warning_upsampling "${phrase_pbmodel_nolm_mono_8k}"

  phrase_pbmodel_withlm_mono_8k=$(deepspeech --model ${TASKCLUSTER_TMP_DIR}/${model_name_mmap} --alphabet ${TASKCLUSTER_TMP_DIR}/alphabet.txt --lm ${TASKCLUSTER_TMP_DIR}/lm.binary --trie ${TASKCLUSTER_TMP_DIR}/trie --audio ${TASKCLUSTER_TMP_DIR}/LDC93S1_pcms16le_1_8000.wav 2>&1 1>/dev/null)
  assert_correct_warning_upsampling "${phrase_pbmodel_withlm_mono_8k}"

  if [ "${aot_model}" = "--aot" ]; then
      phrase_somodel_nolm=$(deepspeech --model "" --alphabet ${TASKCLUSTER_TMP_DIR}/alphabet.txt --audio ${TASKCLUSTER_TMP_DIR}/LDC93S1.wav)
      phrase_somodel_withlm=$(deepspeech --model "" --alphabet ${TASKCLUSTER_TMP_DIR}/alphabet.txt --lm ${TASKCLUSTER_TMP_DIR}/lm.binary --trie ${TASKCLUSTER_TMP_DIR}/trie --audio ${TASKCLUSTER_TMP_DIR}/LDC93S1.wav)

      assert_correct_ldc93s1_somodel "${phrase_somodel_nolm}" "${phrase_somodel_withlm}"

      phrase_somodel_nolm_stereo_44k=$(deepspeech --model "" --alphabet ${TASKCLUSTER_TMP_DIR}/alphabet.txt --audio ${TASKCLUSTER_TMP_DIR}/LDC93S1_pcms16le_2_44100.wav)
      phrase_somodel_withlm_stereo_44k=$(deepspeech --model "" --alphabet ${TASKCLUSTER_TMP_DIR}/alphabet.txt --lm ${TASKCLUSTER_TMP_DIR}/lm.binary --trie ${TASKCLUSTER_TMP_DIR}/trie --audio ${TASKCLUSTER_TMP_DIR}/LDC93S1_pcms16le_2_44100.wav)

      assert_correct_ldc93s1_somodel "${phrase_somodel_nolm_stereo_44k}" "${phrase_somodel_withlm_stereo_44k}"

      phrase_somodel_nolm_mono_8k=$(deepspeech --model "" --alphabet ${TASKCLUSTER_TMP_DIR}/alphabet.txt --audio ${TASKCLUSTER_TMP_DIR}/LDC93S1_pcms16le_1_8000.wav 2>&1 1>/dev/null)
      phrase_somodel_withlm_stereo_44k=$(deepspeech --model "" --alphabet ${TASKCLUSTER_TMP_DIR}/alphabet.txt --lm ${TASKCLUSTER_TMP_DIR}/lm.binary --trie ${TASKCLUSTER_TMP_DIR}/trie --audio ${TASKCLUSTER_TMP_DIR}/LDC93S1_pcms16le_1_8000.wav 2>&1 1>/dev/null)

      assert_correct_warning_upsampling "${phrase_somodel_nolm_mono_8k}" "${phrase_somodel_withlm_mono_8k}"
  fi;
}

run_prod_inference_tests()
{
  phrase_pbmodel_withlm=$(deepspeech --model ${TASKCLUSTER_TMP_DIR}/${model_name} --alphabet ${TASKCLUSTER_TMP_DIR}/alphabet.txt --lm ${TASKCLUSTER_TMP_DIR}/lm.binary --trie ${TASKCLUSTER_TMP_DIR}/trie --audio ${TASKCLUSTER_TMP_DIR}/LDC93S1.wav)
  assert_correct_ldc93s1_prodmodel_v1 "${phrase_pbmodel_withlm}"

  phrase_pbmodel_withlm=$(deepspeech --model ${TASKCLUSTER_TMP_DIR}/${model_name_mmap} --alphabet ${TASKCLUSTER_TMP_DIR}/alphabet.txt --lm ${TASKCLUSTER_TMP_DIR}/lm.binary --trie ${TASKCLUSTER_TMP_DIR}/trie --audio ${TASKCLUSTER_TMP_DIR}/LDC93S1.wav)
  assert_correct_ldc93s1_prodmodel_v2 "${phrase_pbmodel_withlm}"

  phrase_pbmodel_withlm_stereo_44k=$(deepspeech --model ${TASKCLUSTER_TMP_DIR}/${model_name_mmap} --alphabet ${TASKCLUSTER_TMP_DIR}/alphabet.txt --lm ${TASKCLUSTER_TMP_DIR}/lm.binary --trie ${TASKCLUSTER_TMP_DIR}/trie --audio ${TASKCLUSTER_TMP_DIR}/LDC93S1_pcms16le_2_44100.wav)
  assert_working_ldc93s1_prodmodel "${phrase_pbmodel_withlm_stereo_44k}"

  phrase_pbmodel_withlm_mono_8k=$(deepspeech --model ${TASKCLUSTER_TMP_DIR}/${model_name_mmap} --alphabet ${TASKCLUSTER_TMP_DIR}/alphabet.txt --lm ${TASKCLUSTER_TMP_DIR}/lm.binary --trie ${TASKCLUSTER_TMP_DIR}/trie --audio ${TASKCLUSTER_TMP_DIR}/LDC93S1_pcms16le_1_8000.wav 2>&1 1>/dev/null)
  assert_correct_warning_upsampling "${phrase_pbmodel_withlm_mono_8k}"
}

run_multi_inference_tests()
{
  multi_phrase_pbmodel_nolm=$(deepspeech --model ${TASKCLUSTER_TMP_DIR}/${model_name} --alphabet ${TASKCLUSTER_TMP_DIR}/alphabet.txt --audio ${TASKCLUSTER_TMP_DIR}/ | tr '\n' '%')
  assert_correct_multi_ldc93s1 "${multi_phrase_pbmodel_nolm}"

  multi_phrase_pbmodel_withlm=$(deepspeech --model ${TASKCLUSTER_TMP_DIR}/${model_name} --alphabet ${TASKCLUSTER_TMP_DIR}/alphabet.txt --lm ${TASKCLUSTER_TMP_DIR}/lm.binary --trie ${TASKCLUSTER_TMP_DIR}/trie --audio ${TASKCLUSTER_TMP_DIR}/ | tr '\n' '%')
  assert_correct_multi_ldc93s1 "${multi_phrase_pbmodel_withlm}"
}

generic_download_tarxz()
{
  target_dir=$1
  url=$2

  if [ -z "${target_dir}" -o -z "${url}" ]; then
    echo "Empty name for target directory or URL:"
    echo " target_dir=${target_dir}"
    echo " url=${url}"
    exit 1
  fi;

  mkdir -p ${target_dir} || true

  wget ${url} -O - | pixz -d | tar -C ${target_dir} -xf -
}

download_native_client_files()
{
  generic_download_tarxz "$1" "${DEEPSPEECH_ARTIFACTS_ROOT}/native_client.tar.xz"
}

download_aot_model_files()
{
  generic_download_tarxz "$1" "${DEEPSPEECH_AOT_ARTIFACTS_ROOT}/native_client.tar.xz"
}

download_ctc_kenlm()
{
  generic_download_tarxz "$1" "${DEEPSPEECH_LIBCTC}"
}

download_data()
{
  wget -P "${TASKCLUSTER_TMP_DIR}" "${model_source}"
  wget -P "${TASKCLUSTER_TMP_DIR}" "${model_source_mmap}"
  cp ${DS_ROOT_TASK}/DeepSpeech/ds/data/smoke_test/*.wav ${TASKCLUSTER_TMP_DIR}/
  cp ${DS_ROOT_TASK}/DeepSpeech/ds/data/alphabet.txt ${TASKCLUSTER_TMP_DIR}/alphabet.txt
  cp ${DS_ROOT_TASK}/DeepSpeech/ds/data/smoke_test/vocab.pruned.lm ${TASKCLUSTER_TMP_DIR}/lm.binary
  cp ${DS_ROOT_TASK}/DeepSpeech/ds/data/smoke_test/vocab.trie ${TASKCLUSTER_TMP_DIR}/trie
}

download_for_frozen()
{
  wget -O "${TASKCLUSTER_TMP_DIR}/frozen_model.pb" "${DEEPSPEECH_TEST_MODEL}"
}

download_material()
{
  target_dir=$1
  maybe_aot=$2

  if [ "${maybe_aot}" = "--aot" ]; then
    download_aot_model_files "${target_dir}"
  else
    download_native_client_files "${target_dir}"
  fi

  download_data

  ls -hal ${TASKCLUSTER_TMP_DIR}/${model_name} ${TASKCLUSTER_TMP_DIR}/${model_name_mmap} ${TASKCLUSTER_TMP_DIR}/LDC93S1*.wav ${TASKCLUSTER_TMP_DIR}/alphabet.txt
}

install_pyenv()
{
  if [ -z "${PYENV_ROOT}" ]; then
    echo "No PYENV_ROOT set";
    exit 1;
  fi;

  git clone --quiet https://github.com/pyenv/pyenv.git ${PYENV_ROOT}
  pushd ${PYENV_ROOT}
    git checkout --quiet c057a80c8296a7c694e4ef80ecbac0d0c169df7a
  popd
  eval "$(pyenv init -)"
}

install_pyenv_virtualenv()
{
  PYENV_VENV=$1

  if [ -z "${PYENV_VENV}" ]; then
    echo "No PYENV_VENV set";
    exit 1;
  fi;

  git clone --quiet https://github.com/pyenv/pyenv-virtualenv.git ${PYENV_VENV}
  pushd ${PYENV_VENV}
      git checkout --quiet 5419dc732066b035a28680475acd7b661c7c397d
  popd
  eval "$(pyenv virtualenv-init -)"
}

maybe_install_xldd()
{
  # -s required to avoid the noisy output like "Entering / Leaving directories"
  toolchain=$(make -s -C ${DS_DSDIR}/native_client/ TARGET=${SYSTEM_TARGET} TFDIR=${DS_TFDIR} print-toolchain)
  if [ ! -x "${toolchain}ldd" ]; then
    cp "${DS_DSDIR}/native_client/xldd" "${toolchain}ldd" && chmod +x "${toolchain}ldd"
  fi
}

do_get_model_parameters()
{
  local __result=$2
  model_url=$1
  model_file=/tmp/$(basename "${model_url}")

  if [ -z "${model_url}" ]; then
    echo "Empty URL for model"
    exit 1
  fi;

  wget "${model_url}" -O "${model_file}"
  wget -P "/tmp/" "${SUMMARIZE_GRAPH_BINARY}" && chmod +x /tmp/summarize_graph

  if [ ! -f "${model_file}" ]; then
    echo "No such model: ${model_file}"
    exit 1
  fi;

  model_width=$(/tmp/summarize_graph --in_graph="${model_file}" | grep "inputs" | grep -Eo "shape=\[\?,\?,[[:digit:]]+" | cut -d',' -f3)

  eval $__result="'--define=DS_MODEL_FRAMESIZE=${model_width} --define=DS_MODEL_FILE=${model_file}'"
}

# Checks whether we run a patched version of bazel.
# Patching is required to dump computeKey() parameters to .ckd files
# See bazel.patch
# Return 0 (success exit code) on patched version, 1 on release version
is_patched_bazel()
{
  bazel_version=$(bazel version | grep 'Build label:' | cut -d':' -f2)

  if [ -z "${bazel_version}" ]; then
    return 0;
  else
    return 1;
  fi;
}

verify_bazel_rebuild()
{
  bazel_explain_file="$1"

  if [ ! -f "${bazel_explain_file}" ]; then
    echo "No such explain file: ${bazel_explain_file}"
    exit 1
  fi;

  mkdir -p ${TASKCLUSTER_ARTIFACTS} || true

  cp ${DS_ROOT_TASK}/DeepSpeech/tf/bazel*.log ${TASKCLUSTER_ARTIFACTS}/

  spurious_rebuilds=$(grep 'Executing action' "${bazel_explain_file}" | grep 'Compiling' | grep -v -E 'no entry in the cache|unconditional execution is requested' | wc -l)
  if [ "${spurious_rebuilds}" -ne 0 ]; then
    echo "Bazel rebuilds some file it should not, please check."

    if is_patched_bazel; then
      mkdir -p ${DS_ROOT_TASK}/DeepSpeech/ckd/ds ${DS_ROOT_TASK}/DeepSpeech/ckd/tf
      tar xf ${DS_ROOT_TASK}/DeepSpeech/bazel-ckd-tf.tar --strip-components=4 -C ${DS_ROOT_TASK}/DeepSpeech/ckd/ds/
      tar xf ${DS_ROOT_TASK}/DeepSpeech/bazel-ckd-ds.tar --strip-components=4 -C ${DS_ROOT_TASK}/DeepSpeech/ckd/tf/

      echo "Making a diff between CKD files"
      mkdir -p ${TASKCLUSTER_ARTIFACTS}
      diff -urNw ${DS_ROOT_TASK}/DeepSpeech/ckd/tf/ ${DS_ROOT_TASK}/DeepSpeech/ckd/ds/ | tee ${TASKCLUSTER_ARTIFACTS}/ckd.diff

      rm -fr ${DS_ROOT_TASK}/DeepSpeech/ckd/tf/ ${DS_ROOT_TASK}/DeepSpeech/ckd/ds/
    else
      echo "Cannot get CKD information from release, please use patched Bazel"
    fi;

    exit 1
  fi;
}

do_bazel_build()
{
  cd ${DS_ROOT_TASK}/DeepSpeech/tf
  eval "export ${BAZEL_ENV_FLAGS}"

  if is_patched_bazel; then
    find ${DS_ROOT_TASK}/DeepSpeech/tf/bazel-out/ -iname "*.ckd" | tar -cf ${DS_ROOT_TASK}/DeepSpeech/bazel-ckd-tf.tar -T -
  fi;

  bazel ${BAZEL_OUTPUT_USER_ROOT} build \
    -s --explain bazel_monolithic.log --verbose_explanations --experimental_strict_action_env --config=monolithic -c opt ${BAZEL_BUILD_FLAGS} ${BAZEL_TARGETS}

  if is_patched_bazel; then
    find ${DS_ROOT_TASK}/DeepSpeech/tf/bazel-out/ -iname "*.ckd" | tar -cf ${DS_ROOT_TASK}/DeepSpeech/bazel-ckd-ds.tar -T -
  fi;

  verify_bazel_rebuild "${DS_ROOT_TASK}/DeepSpeech/tf/bazel_monolithic.log"
}

do_bazel_shared_build()
{
  cd ${DS_ROOT_TASK}/DeepSpeech/tf
  eval "export ${BAZEL_ENV_FLAGS}"
  bazel ${BAZEL_OUTPUT_USER_ROOT} build \
    -s --explain bazel_shared.log --verbose_explanations --experimental_strict_action_env -c opt ${BAZEL_BUILD_FLAGS} ${BAZEL_TARGETS}
}

do_deepspeech_binary_build()
{
  cd ${DS_DSDIR}
  make -C native_client/ \
    TARGET=${SYSTEM_TARGET} \
    TFDIR=${DS_TFDIR} \
    RASPBIAN=${SYSTEM_RASPBIAN} \
    EXTRA_CFLAGS="${EXTRA_LOCAL_CFLAGS}" \
    EXTRA_LDFLAGS="${EXTRA_LOCAL_LDFLAGS}" \
    EXTRA_LIBS="${EXTRA_LOCAL_LIBS}" \
    deepspeech
}

# Hack to extract Ubuntu's 16.04 libssl 1.0.2 packages and use them during the
# local build of Python.
#
# Avoid (risky) upgrade of base system, allowing to keep one task build that
# builds all the python packages
maybe_ssl102_py37()
{
    pyver=$1

    unset PY37_OPENSSL
    unset PY37_LDPATH
    unset PY37_SOURCE_PACKAGE

    case "${pyver}" in
        3.7*)
            if [ "${OS}" = "Linux" ]; then
                PY37_OPENSSL_DIR=${DS_ROOT_TASK}/ssl-xenial
                mkdir -p ${PY37_OPENSSL_DIR}
                wget -P ${TASKCLUSTER_TMP_DIR} \
                        http://${TASKCLUSTER_WORKER_GROUP}.ec2.archive.ubuntu.com/ubuntu/pool/main/o/openssl/libssl-dev_1.0.2g-1ubuntu4.13_amd64.deb \
                        http://${TASKCLUSTER_WORKER_GROUP}.ec2.archive.ubuntu.com/ubuntu/pool/main/o/openssl/libssl1.0.0_1.0.2g-1ubuntu4.13_amd64.deb

                for deb in ${TASKCLUSTER_TMP_DIR}/libssl*.deb; do
                    dpkg -x ${deb} ${PY37_OPENSSL_DIR}
                done;

                # Python configure expects things to be under lib/
                mv ${PY37_OPENSSL_DIR}/usr/include/x86_64-linux-gnu/openssl/opensslconf.h ${PY37_OPENSSL_DIR}/usr/include/openssl/
                mv ${PY37_OPENSSL_DIR}/lib/x86_64-linux-gnu/lib* ${PY37_OPENSSL_DIR}/usr/lib/
                mv ${PY37_OPENSSL_DIR}/usr/lib/x86_64-linux-gnu/* ${PY37_OPENSSL_DIR}/usr/lib/
                ln -sfn libcrypto.so.1.0.0 ${PY37_OPENSSL_DIR}/usr/lib/libcrypto.so
                ln -sfn libssl.so.1.0.0 ${PY37_OPENSSL_DIR}/usr/lib/libssl.so

                export PY37_OPENSSL="--with-openssl=${PY37_OPENSSL_DIR}/usr"
                export PY37_LDPATH="${PY37_OPENSSL_DIR}/usr/lib/"
            fi;

	    export NUMPY_BUILD_VERSION="==1.14.5"
	    export NUMPY_DEP_VERSION=">=1.14.5"
        ;;
    esac
}

do_deepspeech_python_build()
{
  rename_to_gpu=$1

  unset PYTHON_BIN_PATH
  unset PYTHONPATH
  export PYENV_ROOT="${DS_ROOT_TASK}/DeepSpeech/.pyenv"
  export PATH="${PYENV_ROOT}/bin:$PATH"

  install_pyenv "${PYENV_ROOT}"
  install_pyenv_virtualenv "$(pyenv root)/plugins/pyenv-virtualenv"

  mkdir -p wheels

  SETUP_FLAGS=""
  if [ "${rename_to_gpu}" ]; then
    SETUP_FLAGS="--project_name deepspeech-gpu"
  fi

  for pyver_conf in ${SUPPORTED_PYTHON_VERSIONS}; do
    pyver=$(echo "${pyver_conf}" | cut -d':' -f1)
    pyconf=$(echo "${pyver_conf}" | cut -d':' -f2)

    export NUMPY_BUILD_VERSION="==1.7.0"
    export NUMPY_DEP_VERSION=">=1.7.0"

    maybe_ssl102_py37 ${pyver}

    LD_LIBRARY_PATH=${PY37_LDPATH}:$LD_LIBRARY_PATH PYTHON_CONFIGURE_OPTS="--enable-unicode=${pyconf} ${PY37_OPENSSL}" pyenv install ${pyver}

    pyenv virtualenv ${pyver} deepspeech
    source ${PYENV_ROOT}/versions/${pyver}/envs/deepspeech/bin/activate

    # Set LD path because python ssl might require it
    LD_LIBRARY_PATH=${PY37_LDPATH}:$LD_LIBRARY_PATH \
    EXTRA_CFLAGS="${EXTRA_LOCAL_CFLAGS}" \
    EXTRA_LDFLAGS="${EXTRA_LOCAL_LDFLAGS}" \
    EXTRA_LIBS="${EXTRA_LOCAL_LIBS}" \
    make -C native_client/ \
        TARGET=${SYSTEM_TARGET} \
        RASPBIAN=${SYSTEM_RASPBIAN} \
        TFDIR=${DS_TFDIR} \
        SETUP_FLAGS="${SETUP_FLAGS}" \
        bindings-clean bindings

    cp native_client/dist/*.whl wheels

    make -C native_client/ bindings-clean

    unset NUMPY_BUILD_VERSION
    unset NUMPY_DEP_VERSION

    deactivate
    pyenv uninstall --force deepspeech
    pyenv uninstall --force ${pyver}
  done;
}

do_deepspeech_nodejs_build()
{
  rename_to_gpu=$1

  npm update && npm install node-gyp node-pre-gyp

  export PATH="$(npm root)/.bin/:$PATH"

  for node in ${SUPPORTED_NODEJS_VERSIONS}; do
    EXTRA_CFLAGS="${EXTRA_LOCAL_CFLAGS}" EXTRA_LDFLAGS="${EXTRA_LOCAL_LDFLAGS}" EXTRA_LIBS="${EXTRA_LOCAL_LIBS}" make -C native_client/javascript \
      TARGET=${SYSTEM_TARGET} \
      RASPBIAN=${SYSTEM_RASPBIAN} \
      TFDIR=${DS_TFDIR} \
      NODE_ABI_TARGET=--target=$node \
      clean node-wrapper
  done;

  if [ "${rename_to_gpu}" ]; then
    make -C native_client/javascript clean npm-pack PROJECT_NAME=deepspeech-gpu
  else
    make -C native_client/javascript clean npm-pack
  fi

  tar -czf native_client/javascript/wrapper.tar.gz \
    -C native_client/javascript/ lib/
}

do_deepspeech_npm_package()
{
  cd ${DS_DSDIR}

  npm update && npm install node-gyp node-pre-gyp

  export PATH="$(npm root)/.bin/:$PATH"

  all_tasks="$(curl -s https://queue.taskcluster.net/v1/task/${TASK_ID} | python -c 'import json; import sys; print(" ".join(json.loads(sys.stdin.read())["dependencies"]));')"

  for dep in ${all_tasks}; do
    curl -L https://queue.taskcluster.net/v1/task/${dep}/artifacts/public/wrapper.tar.gz | tar -C native_client/javascript -xzvf -
  done;

  make -C native_client/javascript clean npm-pack
}

package_native_client()
{
  tensorflow_dir=${DS_TFDIR}
  deepspeech_dir=${DS_DSDIR}
  artifacts_dir=${TASKCLUSTER_ARTIFACTS}
  artifact_name=$1

  if [ ! -d ${tensorflow_dir} -o ! -d ${deepspeech_dir} -o ! -d ${artifacts_dir} ]; then
    echo "Missing directory. Please check:"
    echo "tensorflow_dir=${tensorflow_dir}"
    echo "deepspeech_dir=${deepspeech_dir}"
    echo "artifacts_dir=${artifacts_dir}"
    exit 1
  fi;

  if [ -z "${artifact_name}" ]; then
    echo "Please specify artifact name."
  fi;

  if [ -f "${tensorflow_dir}/bazel-bin/native_client/libdeepspeech_model.so" ]; then
    tar -cf - \
      -C ${tensorflow_dir}/bazel-bin/native_client/ generate_trie \
      -C ${tensorflow_dir}/bazel-bin/native_client/ libctc_decoder_with_kenlm.so \
      -C ${tensorflow_dir}/bazel-bin/native_client/ libdeepspeech.so \
      -C ${tensorflow_dir}/bazel-bin/native_client/ libdeepspeech_model.so \
      -C ${tensorflow_dir}/bazel-bin/native_client/ libdeepspeech_utils.so \
      -C ${deepspeech_dir}/ LICENSE \
      -C ${deepspeech_dir}/native_client/ deepspeech \
      -C ${deepspeech_dir}/native_client/kenlm/ README.mozilla \
      | pixz -9 > "${artifacts_dir}/${artifact_name}"
  else
    tar -cf - \
      -C ${tensorflow_dir}/bazel-bin/native_client/ generate_trie \
      -C ${tensorflow_dir}/bazel-bin/native_client/ libctc_decoder_with_kenlm.so \
      -C ${tensorflow_dir}/bazel-bin/native_client/ libdeepspeech.so \
      -C ${tensorflow_dir}/bazel-bin/native_client/ libdeepspeech_utils.so \
      -C ${deepspeech_dir}/ LICENSE \
      -C ${deepspeech_dir}/native_client/ deepspeech \
      -C ${deepspeech_dir}/native_client/kenlm/ README.mozilla \
      | pixz -9 > "${artifacts_dir}/${artifact_name}"
  fi;
}
