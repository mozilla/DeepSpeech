#!/bin/bash

set -xe

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

  ${WGET} ${url} -O - | ${UNXZ} | ${TAR} -C ${target_dir} -xf -
}

generic_download_targz()
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

  ${WGET} ${url} -O - | ${UNGZ} | ${TAR} -C ${target_dir} -xf -
}

download_native_client_files()
{
  local _target_dir=$1
  local _nc_url=$(get_dependency_url "native_client.tar.xz")

  generic_download_tarxz "${_target_dir}" "${_nc_url}"
}

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

get_dependency_url()
{
  local _file=$1
  all_deps="$(curl -s https://community-tc.services.mozilla.com/api/queue/v1/task/${TASK_ID} | python -c 'import json; import sys; print(" ".join(json.loads(sys.stdin.read())["dependencies"]));')"

  for dep in ${all_deps}; do
    local has_artifact=$(curl -s https://community-tc.services.mozilla.com/api/queue/v1/task/${dep}/artifacts | python -c 'import json; import sys; has_artifact = True in [ e["name"].find("'${_file}'") > 0 for e in json.loads(sys.stdin.read())["artifacts"] ]; print(has_artifact)')
    if [ "${has_artifact}" = "True" ]; then
      echo "https://community-tc.services.mozilla.com/api/queue/v1/task/${dep}/artifacts/public/${_file}"
      exit 0
    fi;
  done;

  echo ""
}

download_dependency_file()
{
  local _file=$1
  url=$(get_dependency_url "${_file}")

  if [ -z "${url}" ]; then
    echo "Unable to find an URL for ${_file}"
    exit 1
  fi;

  ${WGET} -P "${TASKCLUSTER_TMP_DIR}" "${url}"
}

download_data()
{
  ${WGET} -P "${TASKCLUSTER_TMP_DIR}" "${model_source}"
  ${WGET} -P "${TASKCLUSTER_TMP_DIR}" "${model_source_mmap}"
  cp ${DS_ROOT_TASK}/DeepSpeech/ds/data/smoke_test/*.wav ${TASKCLUSTER_TMP_DIR}/
  cp ${DS_ROOT_TASK}/DeepSpeech/ds/data/smoke_test/pruned_lm.scorer ${TASKCLUSTER_TMP_DIR}/kenlm.scorer
  cp -R ${DS_ROOT_TASK}/DeepSpeech/ds/native_client/test ${TASKCLUSTER_TMP_DIR}/test_sources
}

download_material()
{
  target_dir=$1

  download_native_client_files "${target_dir}"
  download_data

  ls -hal ${TASKCLUSTER_TMP_DIR}/${model_name} ${TASKCLUSTER_TMP_DIR}/${model_name_mmap} ${TASKCLUSTER_TMP_DIR}/LDC93S1*.wav
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

  mkdir -p ${TASKCLUSTER_ARTIFACTS} || true

  cp ${DS_ROOT_TASK}/DeepSpeech/ds/tensorflow/bazel*.log ${TASKCLUSTER_ARTIFACTS}/

  spurious_rebuilds=$(grep 'Executing action' "${bazel_explain_file}" | grep 'Compiling' | grep -v -E 'no entry in the cache|[for host]|unconditional execution is requested|Executing genrule //native_client:workspace_status|Compiling native_client/workspace_status.cc|Linking native_client/libdeepspeech.so' | wc -l)
  if [ "${spurious_rebuilds}" -ne 0 ]; then
    echo "Bazel rebuilds some file it should not, please check."

    if is_patched_bazel; then
      mkdir -p ${DS_ROOT_TASK}/DeepSpeech/ckd/ds ${DS_ROOT_TASK}/DeepSpeech/ckd/tf
      tar xf ${DS_ROOT_TASK}/DeepSpeech/bazel-ckd-tf.tar --strip-components=4 -C ${DS_ROOT_TASK}/DeepSpeech/ckd/ds/
      tar xf ${DS_ROOT_TASK}/DeepSpeech/bazel-ckd-ds.tar --strip-components=4 -C ${DS_ROOT_TASK}/DeepSpeech/ds/ckd/tensorflow/

      echo "Making a diff between CKD files"
      mkdir -p ${TASKCLUSTER_ARTIFACTS}
      diff -urNw ${DS_ROOT_TASK}/DeepSpeech/ds/ckd/tensorflow/ ${DS_ROOT_TASK}/DeepSpeech/ckd/ds/ | tee ${TASKCLUSTER_ARTIFACTS}/ckd.diff

      rm -fr ${DS_ROOT_TASK}/DeepSpeech/ds/ckd/tensorflow/ ${DS_ROOT_TASK}/DeepSpeech/ckd/ds/
    else
      echo "Cannot get CKD information from release, please use patched Bazel"
    fi;

    exit 1
  fi;
}
