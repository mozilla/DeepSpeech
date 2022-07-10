#!/bin/bash

set -xe

set_ldc_sample_filename()
{
  local _bitrate=$1

  if [ -z "${_bitrate}" ]; then
    echo "Bitrate should not be empty"
    exit 1
  fi;

  case "${_bitrate}" in
    8k)
      ldc93s1_sample_filename='LDC93S1_pcms16le_1_8000.wav'
    ;;
    16k)
      ldc93s1_sample_filename='LDC93S1_pcms16le_1_16000.wav'
    ;;
  esac
}

download_model_prod()
{
  local _model_source_file=$(basename "${model_source}")
  ${WGET} "${model_source}" -O - | gunzip --force > "${CI_TMP_DIR}/${_model_source_file}"

  local _model_source_mmap_file=$(basename "${model_source_mmap}")
  ${WGET} "${model_source_mmap}" -O - | gunzip --force > "${CI_TMP_DIR}/${_model_source_mmap_file}"
}

download_data()
{
  cp ${DS_DSDIR}/data/smoke_test/*.wav ${CI_TMP_DIR}/
  cp ${DS_DSDIR}/data/smoke_test/pruned_lm.scorer ${CI_TMP_DIR}/kenlm.scorer
  cp ${DS_DSDIR}/data/smoke_test/pruned_lm.bytes.scorer ${CI_TMP_DIR}/kenlm.bytes.scorer

  cp -R ${DS_DSDIR}/native_client/test ${CI_TMP_DIR}/test_sources
}

download_material()
{
  download_data

  ls -hal ${CI_TMP_DIR}/${model_name} ${CI_TMP_DIR}/${model_name_mmap} ${CI_TMP_DIR}/LDC93S1*.wav
}

maybe_install_xldd()
{
  # -s required to avoid the noisy output like "Entering / Leaving directories"
  toolchain=$(make -s -C ${DS_DSDIR}/native_client/ TARGET=${SYSTEM_TARGET} TFDIR=${DS_TFDIR} print-toolchain)
  if [ ! -x "${toolchain}ldd" ]; then
    cp "${DS_DSDIR}/native_client/xldd" "${toolchain}ldd" && chmod +x "${toolchain}ldd"
  fi
}

# Checks whether we run a patched version of bazel.
# Patching is required to dump computeKey() parameters to .ckd files
# See bazel.patch
# Return 0 (success exit code) on patched version, 1 on release version
is_patched_bazel()
{
  bazel_version=$(bazel version | grep 'Build label:' | cut -d':' -f2)

  bazel shutdown

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

  mkdir -p ${CI_ARTIFACTS_DIR} || true

  cp ${DS_DSDIR}/tensorflow/bazel*.log ${CI_ARTIFACTS_DIR}/

  spurious_rebuilds=$(grep 'Executing action' "${bazel_explain_file}" | grep 'Compiling' | grep -v -E 'no entry in the cache|[for host]|unconditional execution is requested|Executing genrule //native_client:workspace_status|Compiling native_client/workspace_status.cc|Linking native_client/libdeepspeech.so' | wc -l)
  if [ "${spurious_rebuilds}" -ne 0 ]; then
    echo "Bazel rebuilds some file it should not, please check."

    if is_patched_bazel; then
      mkdir -p ${DS_ROOT_TASK}/ckd/ds ${DS_ROOT_TASK}/ckd/tf
      tar xf ${DS_ROOT_TASK}/bazel-ckd-tf.tar --strip-components=4 -C ${DS_ROOT_TASK}/ckd/ds/
      tar xf ${DS_ROOT_TASK}/bazel-ckd-ds.tar --strip-components=4 -C ${DS_DSDIR}/ckd/tensorflow/

      echo "Making a diff between CKD files"
      mkdir -p ${CI_ARTIFACTS_DIR}
      diff -urNw ${DS_DSDIR}/ckd/tensorflow/ ${DS_ROOT_TASK}/ckd/ds/ | tee ${CI_ARTIFACTS_DIR}/ckd.diff

      rm -fr ${DS_DSDIR}/ckd/tensorflow/ ${DS_ROOT_TASK}/ckd/ds/
    else
      echo "Cannot get CKD information from release, please use patched Bazel"
    fi;

    exit 1
  fi;
}

symlink_electron()
{
  if [ "${OS}" = "Darwin" ]; then
    ln -s Electron.app/Contents/MacOS/Electron node_modules/electron/dist/node
  else
    ln -s electron "${DS_ROOT_TASK}/node_modules/electron/dist/node"

    if [ "${OS}" = "Linux" -a -f "${DS_ROOT_TASK}/node_modules/electron/dist/chrome-sandbox" ]; then
      export ELECTRON_DISABLE_SANDBOX=1
    fi
  fi
}

export_node_bin_path()
{
  export PATH=${DS_ROOT_TASK}/node_modules/.bin/:${DS_ROOT_TASK}/node_modules/electron/dist/:$PATH
}

export_py_bin_path()
{
  export PATH=$HOME/.local/bin/:$PATH
}
