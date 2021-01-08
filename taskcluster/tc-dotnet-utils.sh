#!/bin/bash

set -xe

install_nuget()
{
  PROJECT_NAME=$1
  if [ -z "${PROJECT_NAME}" ]; then
    exit "Please call with a valid PROJECT_NAME"
    exit 1
  fi;

  nuget="${PROJECT_NAME}.${DS_VERSION}.nupkg"

  export PATH=$PATH:$(cygpath ${ChocolateyInstall})/bin

  mkdir -p "${TASKCLUSTER_TMP_DIR}/repo/"
  mkdir -p "${TASKCLUSTER_TMP_DIR}/ds/"

  nuget_pkg_url=$(get_dependency_url "${nuget}")
  console_pkg_url=$(get_dependency_url "DeepSpeechConsole.exe")

  ${WGET} -O - "${nuget_pkg_url}" | gunzip > "${TASKCLUSTER_TMP_DIR}/${PROJECT_NAME}.${DS_VERSION}.nupkg"
  ${WGET} -O - "${console_pkg_url}" | gunzip > "${TASKCLUSTER_TMP_DIR}/ds/DeepSpeechConsole.exe"

  nuget sources add -Name repo -Source $(cygpath -w "${TASKCLUSTER_TMP_DIR}/repo/")

  cd "${TASKCLUSTER_TMP_DIR}"
  nuget add $(cygpath -w "${TASKCLUSTER_TMP_DIR}/${nuget}") -source repo

  cd "${TASKCLUSTER_TMP_DIR}/ds/"
  nuget list -Source repo -Prerelease
  nuget install ${PROJECT_NAME} -Source repo -Prerelease

  ls -halR "${PROJECT_NAME}.${DS_VERSION}"

  nuget install NAudio
  cp NAudio*/lib/net35/NAudio.dll ${TASKCLUSTER_TMP_DIR}/ds/
  cp ${PROJECT_NAME}.${DS_VERSION}/build/libdeepspeech.so ${TASKCLUSTER_TMP_DIR}/ds/
  cp ${PROJECT_NAME}.${DS_VERSION}/lib/net46/DeepSpeechClient.dll ${TASKCLUSTER_TMP_DIR}/ds/

  ls -hal ${TASKCLUSTER_TMP_DIR}/ds/

  export PATH=${TASKCLUSTER_TMP_DIR}/ds/:$PATH
}
