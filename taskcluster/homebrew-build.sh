#!/bin/bash

set -xe

OS=$(uname)
if [ "${OS}" != "Darwin" ]; then
    echo "This should only run on OSX."
    exit 1
fi;

flavor=$1

source $(dirname "$0")/tc-tests-utils.sh

if [ -z "${TASKCLUSTER_TASK_DIR}" ]; then
  echo "No TASKCLUSTER_TASK_DIR, aborting."
  exit 1
fi

do_prepare_homebrew()
{
  local _brew_instance=$1

  if [ -z "${_brew_instance}" ]; then
    echo "No _brew_instance, aborting."
    exit 1
  fi

  export PATH=${_brew_instance}/bin:$PATH
  export HOMEBREW_LOGS="${_brew_instance}/homebrew.logs/"
  export HOMEBREW_CACHE="${_brew_instance}/homebrew.cache/"
  export BREW_FORMULAS_COMMIT=ddd39cf1b71452bfe9c5f17f45cc0118796b20d3

  # Never fail on pre-existing homebrew/ directory
  mkdir -p "${_brew_instance}" || true
  mkdir -p "${HOMEBREW_CACHE}" || true

  # Make sure to verify there is a 'brew' binary there, otherwise install things.
  if [ ! -x "${_brew_instance}/bin/brew" ]; then
    curl -L ${BREW_URL} | tar xz --strip 1 -C "${_brew_instance}"
  fi;

  check_homebrew "${_brew_instance}"

  # Then we force onto a specific well-known commit
  mkdir -p "$(brew --prefix)/Library/Taps/homebrew/homebrew-core"
  pushd "$(brew --prefix)/Library/Taps/homebrew/homebrew-core"
    git init
    git remote add origin https://github.com/Homebrew/homebrew-core.git
    git fetch origin
    git checkout ${BREW_FORMULAS_COMMIT}
  popd
}

check_homebrew()
{
  local _expected_prefix=$1

  echo "local brew prefix ..."
  local _local_prefix=$(brew --prefix)
  echo "${_local_prefix}"

  if [ "${_expected_prefix}" != "${_local_prefix}" ]; then
    echo "Weird state:"
    echo "_expected_prefix=${_expected_prefix}"
    echo "_local_prefix=${_local_prefix}"
    exit 1
  fi;
}

install_pkg_homebrew()
{
  local pkg=$1
  (brew list --versions ${pkg} && brew upgrade --force-bottle ${pkg}) || brew install --force-bottle ${pkg}
}

prepare_homebrew_builds()
{
  do_prepare_homebrew "${BUILDS_BREW}"

  install_pkg_homebrew "coreutils"
  install_pkg_homebrew "node@12"
  install_pkg_homebrew "openssl"
  install_pkg_homebrew "pkg-config"
  install_pkg_homebrew "pyenv-virtualenv"
  install_pkg_homebrew "readline"
  install_pkg_homebrew "sox"
}

prepare_homebrew_tests()
{
  do_prepare_homebrew "${TESTS_BREW}"

  install_pkg_homebrew "nvm"
    source "${TESTS_BREW}/opt/nvm/nvm.sh"
    for node_ver in ${SUPPORTED_NODEJS_TESTS_VERSIONS};
    do
      nvm install ${node_ver}
    done;

  install_pkg_homebrew "openssl"
  install_pkg_homebrew "pkg-config"
  install_pkg_homebrew "readline"
  install_pkg_homebrew "sox"
}

if [ "${flavor}" = "--builds" ]; then
  prepare_homebrew_builds
fi;

if [ "${flavor}" = "--tests" ]; then
  prepare_homebrew_tests
fi;
