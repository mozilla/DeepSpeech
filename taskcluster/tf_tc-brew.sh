#!/bin/bash

set -ex

# if [ -z "${TASKCLUSTER_TASK_DIR}" ]; then
#     echo "No TASKCLUSTER_TASK_DIR, aborting."
#     exit 1
# fi

# LOCAL_BREW="${TASKCLUSTER_TASK_DIR}/homebrew"
# export PATH=${LOCAL_BREW}/bin:$PATH
# export HOMEBREW_LOGS="${TASKCLUSTER_TASK_DIR}/homebrew.logs/"
# export HOMEBREW_CACHE="${TASKCLUSTER_TASK_DIR}/homebrew.cache/"
# export HOMEBREW_FORMULAS_COMMIT=93fe256e0168db3b1c70c26a01941be59ce76311
# export HOMEBREW_NO_AUTO_UPDATE=1

# # Never fail on pre-existing homebrew/ directory
# mkdir -p "${LOCAL_BREW}" || true
# mkdir -p "${HOMEBREW_CACHE}" || true

# # Make sure to verify there is a 'brew' binary there, otherwise install things.
# if [ ! -x "${LOCAL_BREW}/bin/brew" ]; then
#     curl -L https://github.com/Homebrew/brew/tarball/2.2.17 | tar xz --strip 1 -C "${LOCAL_BREW}"
# fi;

# echo "local brew list (should be empty) ..."
# brew list

# echo "local brew prefix ..."
# local_prefix=$(brew --prefix)
# echo "${local_prefix}"

# if [ "${LOCAL_BREW}" != "${local_prefix}" ]; then
#     echo "Weird state:"
#     echo "LOCAL_BREW=${LOCAL_BREW}"
#     echo "local_prefix=${local_prefix}"
#     exit 1
# fi;


# # Then we force onto a specific well-known commit
# mkdir -p "$(brew --prefix)/Library/Taps/homebrew/homebrew-core"
# pushd "$(brew --prefix)/Library/Taps/homebrew/homebrew-core"
#   git init
#   git remote add origin https://github.com/Homebrew/homebrew-core.git
#   git fetch origin
#   git checkout ${HOMEBREW_FORMULAS_COMMIT}
# popd

# # coreutils, pyenv-virtualenv required for build of tensorflow
# all_pkgs="coreutils pyenv-virtualenv"

# for pkg in ${all_pkgs};
# do
#    (brew list --versions ${pkg} && brew upgrade ${pkg}) || brew install ${pkg}
# done;
