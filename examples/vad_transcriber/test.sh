#!/bin/bash

set -xe

THIS=$(dirname "$0")

pushd ${THIS}
  source ../tests.sh

  pip install --user $(get_python_wheel_url "$1")
  pip install --user -r requirements.txt

  python audioTranscript_cmd.py \
	  --audio $HOME/DeepSpeech/audio/2830-3980-0043.wav \
	  --aggressive 0 \
	  --model $HOME/DeepSpeech/models/

  python audioTranscript_cmd.py \
	  --audio $HOME/DeepSpeech/audio/2830-3980-0043.wav \
	  --aggressive 0 \
	  --model $HOME/DeepSpeech/models/ \
	  --stream
popd
