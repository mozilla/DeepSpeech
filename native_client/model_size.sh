#!/bin/sh

# Fail if we don't have *three* arguments
#  $1 => file to generate
#  $2 => dimension to use for timesteps
#  $3 => dimension to use for framesize
if [ $# -ne 3 ]; then
    exit 1
fi;

sed -e "s|\$DS_MODEL_TIMESTEPS|$2|g" -e "s|\$DS_MODEL_FRAMESIZE|$3|g" < $1
