#!/bin/bash

set -xe

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

  ${TAR} -cf - \
    -C ${tensorflow_dir}/bazel-bin/native_client/ libdeepspeech.so \
    -C ${tensorflow_dir}/bazel-bin/native_client/ libdeepspeech.so.if.lib \
    -C ${tensorflow_dir}/bazel-bin/native_client/ generate_scorer_package \
    -C ${deepspeech_dir}/ LICENSE \
    -C ${deepspeech_dir}/native_client/ deepspeech${PLATFORM_EXE_SUFFIX} \
    -C ${deepspeech_dir}/native_client/ deepspeech.h \
    -C ${deepspeech_dir}/native_client/kenlm/ README.mozilla \
    | ${XZ} > "${artifacts_dir}/${artifact_name}"
}

package_native_client_ndk()
{
  deepspeech_dir=${DS_DSDIR}
  tensorflow_dir=${DS_TFDIR}
  artifacts_dir=${TASKCLUSTER_ARTIFACTS}
  artifact_name=$1
  arch_abi=$2

  if [ ! -d ${deepspeech_dir} -o ! -d ${artifacts_dir} ]; then
    echo "Missing directory. Please check:"
    echo "deepspeech_dir=${deepspeech_dir}"
    echo "artifacts_dir=${artifacts_dir}"
    exit 1
  fi;

  if [ -z "${artifact_name}" ]; then
    echo "Please specify artifact name."
  fi;

  if [ -z "${arch_abi}" ]; then
    echo "Please specify arch abi."
  fi;

  tar -cf - \
    -C ${deepspeech_dir}/native_client/libs/${arch_abi}/ deepspeech \
    -C ${deepspeech_dir}/native_client/libs/${arch_abi}/ libdeepspeech.so \
    -C ${tensorflow_dir}/bazel-bin/native_client/ generate_scorer_package \
    -C ${deepspeech_dir}/native_client/libs/${arch_abi}/ libc++_shared.so \
    -C ${deepspeech_dir}/native_client/ deepspeech.h \
    -C ${deepspeech_dir}/ LICENSE \
    -C ${deepspeech_dir}/native_client/kenlm/ README.mozilla \
    | pixz -9 > "${artifacts_dir}/${artifact_name}"
}

package_libdeepspeech_as_zip()
{
  tensorflow_dir=${DS_TFDIR}
  artifacts_dir=${TASKCLUSTER_ARTIFACTS}
  artifact_name=$1

  if [ ! -d ${tensorflow_dir} -o ! -d ${artifacts_dir} ]; then
    echo "Missing directory. Please check:"
    echo "tensorflow_dir=${tensorflow_dir}"
    echo "artifacts_dir=${artifacts_dir}"
    exit 1
  fi;

  if [ -z "${artifact_name}" ]; then
    echo "Please specify artifact name."
  fi;

  zip -r9 --junk-paths "${artifacts_dir}/${artifact_name}" ${tensorflow_dir}/bazel-bin/native_client/libdeepspeech.so
}
