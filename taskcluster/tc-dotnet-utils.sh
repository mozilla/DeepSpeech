#!/bin/bash

set -xe

install_nuget()
{
  PROJECT_NAME=$1
  if [ -z "${PROJECT_NAME}" ]; then
    exit "Please call with a valid PROJECT_NAME"
    exit 1
  fi;
  
  ConsolePkgName=$2
  if [ -z "${ConsolePkgName}" ]; then
    exit "Please call with a valid Console Project name (DeepSpeechConsole.exe or DeepSpeechConsoleNetCore.exe)"
    exit 1
  fi;
  
  nuget="${PROJECT_NAME}.${DS_VERSION}.nupkg"

  export PATH=$PATH:$(cygpath ${ChocolateyInstall})/bin

  mkdir -p "${TASKCLUSTER_TMP_DIR}/repo/"
  mkdir -p "${TASKCLUSTER_TMP_DIR}/ds/"

  nuget_pkg_url=$(get_dep_nuget_pkg_url "${nuget}")
  console_pkg_url=$(get_dep_nuget_pkg_url "${ConsolePkgName}")

  ${WGET} -O - "${nuget_pkg_url}" | gunzip > "${TASKCLUSTER_TMP_DIR}/${PROJECT_NAME}.${DS_VERSION}.nupkg"
  ${WGET} -O - "${console_pkg_url}" | gunzip > "${TASKCLUSTER_TMP_DIR}/ds/${ConsolePkgName}"

  nuget sources add -Name repo -Source $(cygpath -w "${TASKCLUSTER_TMP_DIR}/repo/")

  cd "${TASKCLUSTER_TMP_DIR}"
  nuget add $(cygpath -w "${TASKCLUSTER_TMP_DIR}/${nuget}") -source repo

  cd "${TASKCLUSTER_TMP_DIR}/ds/"
  nuget list -Source repo -Prerelease
  nuget install ${PROJECT_NAME} -Source repo -Prerelease

  ls -halR "${PROJECT_NAME}.${DS_VERSION}"

  nuget install NAudio
  if [ "${ConsolePkgName}" = "DeepSpeechConsole.exe" ]; then
    cp NAudio*/lib/net35/NAudio.dll ${TASKCLUSTER_TMP_DIR}/ds/
    cp ${PROJECT_NAME}.${DS_VERSION}/runtimes/win-x64/native/libdeepspeech.so ${TASKCLUSTER_TMP_DIR}/ds/
    cp ${PROJECT_NAME}.${DS_VERSION}/lib/net46/DeepSpeechClient.dll ${TASKCLUSTER_TMP_DIR}/ds/
  elif [ "${ConsolePkgName}" = "DeepSpeechConsoleNetCore.exe" ]; then
    cp NAudio*/lib/netstandard2.0/NAudio.dll ${TASKCLUSTER_TMP_DIR}/ds/
    cp ${PROJECT_NAME}.${DS_VERSION}/runtimes/win-x64/native/libdeepspeech.so ${TASKCLUSTER_TMP_DIR}/ds/
    cp ${PROJECT_NAME}.${DS_VERSION}/lib/netcoreapp3.1/DeepSpeechClient.dll ${TASKCLUSTER_TMP_DIR}/ds/
  else
    exit "Please call with a valid Console Project name (DeepSpeechConsole.exe or DeepSpeechConsoleNetCore.exe)"
    exit 1
  fi

  ls -hal ${TASKCLUSTER_TMP_DIR}/ds/

  export PATH=${TASKCLUSTER_TMP_DIR}/ds/:$PATH
}

# Will inspect this task's dependencies for one that provides a matching NuGet package
get_dep_nuget_pkg_url()
{
  local deepspeech_pkg=$1
  local all_deps=get_all_deps_from_task

  for dep in ${all_deps}; do
    local has_artifact=$(curl -s https://community-tc.services.mozilla.com/api/queue/v1/task/${dep}/artifacts | python -c 'import json; import sys; has_artifact = True in [ e["name"].find("'${deepspeech_pkg}'") > 0 for e in json.loads(sys.stdin.read())["artifacts"] ]; print(has_artifact)')
    if [ "${has_artifact}" = "True" ]; then
      echo "https://community-tc.services.mozilla.com/api/queue/v1/task/${dep}/artifacts/public/${deepspeech_pkg}"
      exit 0
    fi;
  done;

  echo ""
  # This should not be reached, otherwise it means we could not find a matching nodejs package
  exit 1
}


get_all_deps_from_task()
{
  echo "$(curl -s https://community-tc.services.mozilla.com/api/queue/v1/task/${TASK_ID} | python -c 'import json; import sys; print(" ".join(json.loads(sys.stdin.read())["dependencies"]));')"
  exit 0
}