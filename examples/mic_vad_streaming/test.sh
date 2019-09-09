#!/bin/bash

set -xe

THIS=$(dirname "$0")

pushd ${THIS}
  source ../tests.sh

  pip install --user $(get_python_wheel_url "$1")
  pip install --user -r requirements.txt

  pulseaudio &

  python mic_vad_streaming.py \
	  --model $HOME/DeepSpeech/models/output_graph.pbmm \
	  --alphabet $HOME/DeepSpeech/models/alphabet.txt \
	  --lm $HOME/DeepSpeech/models/lm.binary \
	  --trie $HOME/DeepSpeech/models/trie \
	  --file $HOME/DeepSpeech/audio/2830-3980-0043.wav
popd
