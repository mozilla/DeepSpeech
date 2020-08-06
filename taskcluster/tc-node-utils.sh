#!/bin/bash

set -xe

# Will inspect this task's dependencies for one that provides a matching npm package
get_dep_npm_pkg_url()
{
  local all_deps="$(curl -s https://community-tc.services.mozilla.com/api/queue/v1/task/${TASK_ID} | python -c 'import json; import sys; print(" ".join(json.loads(sys.stdin.read())["dependencies"]));')"

  # We try "mozilla-voice-stt-tflite" and "mozilla-voice-stt-cuda" first and if we don't find it we try "mozilla-voice-stt"
  for pkg_basename in "mozilla-voice-stt-tflite" "mozilla-voice-stt-cuda" "mozilla-voice-stt"; do
    local deepspeech_pkg="${pkg_basename}-${DS_VERSION}.tgz"
    for dep in ${all_deps}; do
      local has_artifact=$(curl -s https://community-tc.services.mozilla.com/api/queue/v1/task/${dep}/artifacts | python -c 'import json; import sys; has_artifact = True in [ e["name"].find("'${deepspeech_pkg}'") > 0 for e in json.loads(sys.stdin.read())["artifacts"] ]; print(has_artifact)')
      if [ "${has_artifact}" = "True" ]; then
        echo "https://community-tc.services.mozilla.com/api/queue/v1/task/${dep}/artifacts/public/${deepspeech_pkg}"
        exit 0
      fi;
    done;
  done;

  echo ""
  # This should not be reached, otherwise it means we could not find a matching nodejs package
  exit 1
}
