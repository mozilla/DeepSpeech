#!/bin/sh

set -xe

python -m readme2tex --svgdir doc/svgs/ --username mozilla --project DeepSpeech --output doc/DeepSpeech.md doc/DeepSpeech.raw
python -m readme2tex --svgdir doc/svgs/ --username mozilla --project DeepSpeech --output doc/Geometry.md doc/Geometry.raw
python -m readme2tex --svgdir doc/svgs/ --username mozilla --project DeepSpeech --output doc/ParallelOptimization.md doc/ParallelOptimization.raw
