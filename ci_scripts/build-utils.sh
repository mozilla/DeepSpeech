#!/bin/bash

set -xe

do_bazel_build()
{
  local _opt_or_dbg=${1:-"opt"}

  cd ${DS_TFDIR}
  eval "export ${BAZEL_ENV_FLAGS}"

  if [ "${_opt_or_dbg}" = "opt" ]; then
    if is_patched_bazel; then
      find ${DS_ROOT_TASK}/tensorflow/bazel-out/ -iname "*.ckd" | tar -cf ${DS_ROOT_TASK}/bazel-ckd-tf.tar -T -
    fi;
  fi;

  bazel ${BAZEL_OUTPUT_USER_ROOT} build \
    -s --explain bazel_monolithic.log --verbose_explanations --experimental_strict_action_env --workspace_status_command="bash native_client/bazel_workspace_status_cmd.sh" --config=monolithic -c ${_opt_or_dbg} ${BAZEL_BUILD_FLAGS} ${BAZEL_TARGETS}

  if [ "${_opt_or_dbg}" = "opt" ]; then
    if is_patched_bazel; then
      find ${DS_ROOT_TASK}/tensorflow/bazel-out/ -iname "*.ckd" | tar -cf ${DS_ROOT_TASK}/bazel-ckd-ds.tar -T -
    fi;
    verify_bazel_rebuild "${DS_ROOT_TASK}/tensorflow/bazel_monolithic.log"
  fi;
}

shutdown_bazel()
{
  cd ${DS_TFDIR}
  bazel ${BAZEL_OUTPUT_USER_ROOT} shutdown
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
    deepspeech${PLATFORM_EXE_SUFFIX}
}
