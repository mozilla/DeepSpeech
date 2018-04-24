#!/bin/bash

set -xe

this_dir=$(dirname "$0")/
NPROC=$(nproc)

cat <<EOF > ${HOME}/.makepkg.conf
MAKEFLAGS="-j${NPROC}"
EOF

cd ${this_dir} && makepkg --noconfirm -sri
