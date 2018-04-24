#!/bin/bash

set -xe

this_dir=$(dirname "$0")/

mkdir -p /tmp/artifacts/

cd ${this_dir} && cp \
	*deepspeech*.pkg.tar.xz \
	/tmp/artifacts/
