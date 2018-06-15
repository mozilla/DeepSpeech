#!/bin/bash

set -xe

OS=$(uname)
if [ "${OS}" != "Darwin" ]; then
    echo "This should only run on OSX."
    exit 1
fi;

install_local_homebrew()
{
    LOCAL_HOMEBREW_PREFIX=$1

    if [ -z "${TASKCLUSTER_TASK_DIR}" ]; then
        echo "No TASKCLUSTER_TASK_DIR, aborting."
        exit 1
    fi

    if [ -z "${LOCAL_HOMEBREW_PREFIX}" ]; then
        echo "No LOCAL_HOMEBREW_PREFIX, aborting."
        exit 1
    fi

    for suffix in .brew .cache .logs;
    do
        if [ -d "${TASKCLUSTER_TASK_DIR}/${LOCAL_HOMEBREW_PREFIX}${suffix}/" ]; then
            echo "Directory ${TASKCLUSTER_TASK_DIR}/${LOCAL_HOMEBREW_PREFIX}${suffix} already exists, aborting"
            exit 1
        fi
    done;

    export LOCAL_HOMEBREW_DIRECTORY="${TASKCLUSTER_TASK_DIR}/${LOCAL_HOMEBREW_PREFIX}.brew"
    export HOMEBREW_LOGS="${TASKCLUSTER_TASK_DIR}/${LOCAL_HOMEBREW_PREFIX}.logs"
    export HOMEBREW_CACHE="${TASKCLUSTER_TASK_DIR}/${LOCAL_HOMEBREW_PREFIX}.cache"

    mkdir -p "${LOCAL_HOMEBREW_DIRECTORY}"
    mkdir -p "${HOMEBREW_CACHE}"

    curl -L https://github.com/Homebrew/brew/tarball/master | tar xz --strip 1 -C "${LOCAL_HOMEBREW_DIRECTORY}"
    export PATH=${LOCAL_HOMEBREW_DIRECTORY}/bin:$PATH

    if [ ! -x "${LOCAL_HOMEBREW_DIRECTORY}/bin/brew" ]; then
        echo "No brew binary under ${LOCAL_HOMEBREW_DIRECTORY}, aborting"
        exit 1
    fi;

    echo "local brew list (should be empty) ..."
    brew list

    echo "local brew prefix ..."
    local_prefix=$(brew --prefix)
    echo "${local_prefix}"

    if [ "${LOCAL_HOMEBREW_DIRECTORY}" != "${local_prefix}" ]; then
        echo "Weird state:"
        echo "LOCAL_HOMEBREW_DIRECTORY=${LOCAL_HOMEBREW_DIRECTORY}"
        echo "local_prefix=${local_prefix}"
        exit 1
    fi;
}

install_pkg_local_homebrew()
{
    pkg=$1

    if [ ! -x "${LOCAL_HOMEBREW_DIRECTORY}/bin/brew" ]; then
        echo "Cannot install $pkg: no brew binary under ${LOCAL_HOMEBREW_DIRECTORY}, aborting"
        exit 1
    fi;

    (brew list --versions ${pkg} && brew upgrade ${pkg}) || brew install ${pkg}
}
