#!/bin/bash

set -xe

install_pyenv()
{
  if [ -z "${PYENV_ROOT}" ]; then
    echo "No PYENV_ROOT set";
    exit 1;
  fi;

  if [ "${OS}" = "${TC_MSYS_VERSION}" ]; then
    mkdir -p "${PYENV_ROOT}/versions/"
    return;
  fi

  # Allows updating local cache if required
  if [ ! -e "${PYENV_ROOT}/bin/pyenv" ]; then
    git clone --quiet https://github.com/pyenv/pyenv.git ${PYENV_ROOT}
  else
    pushd ${PYENV_ROOT}
      git fetch origin
    popd
  fi

  pushd ${PYENV_ROOT}
    git checkout --quiet 20a1f0cd7a3d2f95800d8e0d5863b4e98f25f4df
  popd

  if [ ! -d "${PYENV_ROOT}/plugins/pyenv-alias" ]; then
    git clone https://github.com/s1341/pyenv-alias.git ${PYENV_ROOT}/plugins/pyenv-alias
    pushd ${PYENV_ROOT}/plugins/pyenv-alias
      git checkout --quiet 8896eebb5b47389249b35d21d8a5e74aa33aff08
    popd
  fi

  eval "$(pyenv init -)"
}

install_pyenv_virtualenv()
{
  local PYENV_VENV=$1

  if [ -z "${PYENV_VENV}" ]; then
    echo "No PYENV_VENV set";
    exit 1;
  fi;

  if [ "${OS}" = "${TC_MSYS_VERSION}" ]; then
    echo "No pyenv virtualenv support ; will install virtualenv locally from pip"
    return
  fi;

  if [ ! -e "${PYENV_VENV}/bin/pyenv-virtualenv" ]; then
    git clone --quiet https://github.com/pyenv/pyenv-virtualenv.git ${PYENV_VENV}
    pushd ${PYENV_VENV}
        git checkout --quiet 5419dc732066b035a28680475acd7b661c7c397d
    popd
  fi;

  eval "$(pyenv virtualenv-init -)"
}

maybe_setup_virtualenv_cross_arm()
{
  local version=$1
  local name=$2

  if [ -z "${PYENV_ROOT}" ]; then
    echo "No PYENV_ROOT set";
    exit 1;
  fi;

  if [ "${OS}" != "Linux" ]; then
    echo "Only for Linux/ARM arch";
    return 0;
  fi;

  ARCH=$(uname -m)

  if [ "${ARCH}" = "x86_64" ]; then
    echo "Only for Linux/ARM arch";
    return 0;
  fi;

  mkdir -p ${PYENV_ROOT}/versions/${version}/envs/

  PIP_EXTRA_INDEX_URL="" python3 -m virtualenv -p python3 ${PYENV_ROOT}/versions/${version}/envs/${name}/
  source ${PYENV_ROOT}/versions/${version}/envs/${name}/bin/activate
}

setup_pyenv_virtualenv()
{
  local version=$1
  local name=$2

  if [ -z "${PYENV_ROOT}" ]; then
    echo "No PYENV_ROOT set";
    exit 1;
  fi;

  if [ "${OS}" = "${TC_MSYS_VERSION}" ]; then
    echo "should setup virtualenv ${name} for ${version}"
    mkdir ${PYENV_ROOT}/versions/${version}/envs
    PATH=${PYENV_ROOT}/versions/${version}/tools:${PYENV_ROOT}/versions/${version}/tools/Scripts:$PATH python -m venv ${PYENV_ROOT}/versions/${version}/envs/${name}
  else
    ls -hal "${PYENV_ROOT}/versions/"

    # There could be a symlink when re-using cacche on macOS
    # We don't care, let's just remove it
    if [ -L "${PYENV_ROOT}/versions/${name}" ]; then
      rm "${PYENV_ROOT}/versions/${name}"
    fi

    # Don't force-reinstall existing version
    if [ ! -f "${PYENV_ROOT}/versions/${version}/envs/${name}/bin/activate" ]; then
      pyenv virtualenv ${version} ${name}
    fi
  fi
}

virtualenv_activate()
{
  local version=$1
  local name=$2

  if [ -z "${PYENV_ROOT}" ]; then
    echo "No PYENV_ROOT set";
    exit 1;
  fi;

  if [ "${OS}" = "${TC_MSYS_VERSION}" ]; then
    source ${PYENV_ROOT}/versions/${version}/envs/${name}/Scripts/activate
  else
    source ${PYENV_ROOT}/versions/${version}/envs/${name}/bin/activate
  fi
}

virtualenv_deactivate()
{
  local version=$1
  local name=$2

  if [ -z "${PYENV_ROOT}" ]; then
    echo "No PYENV_ROOT set";
    exit 1;
  fi;

  deactivate
}

pyenv_install()
{
  local version=$1
  local version_alias=$2

  if [ -z "${version_alias}" ]; then
    echo "WARNING, no version_alias specified, please ensure call site is okay"
    version_alias=${version}
  fi;

  if [ -z "${PYENV_ROOT}" ]; then
    echo "No PYENV_ROOT set";
    exit 1;
  fi;

  if [ -z "${DS_CPU_COUNT}" ]; then
    echo "No idea of parallelism";
    exit 1;
  fi;

  if [ "${OS}" = "${TC_MSYS_VERSION}" ]; then
    PATH=$(cygpath ${ChocolateyInstall})/bin:$PATH nuget install python -Version ${version} -OutputDirectory ${PYENV_ROOT}/versions/

    mv ${PYENV_ROOT}/versions/python.${version} ${PYENV_ROOT}/versions/${version_alias}

    PY_TOOLS_DIR="$(cygpath -w ${PYENV_ROOT}/versions/${version_alias}/tools/)"
    TEMP=$(cygpath -w ${DS_ROOT_TASK}/tmp/) PATH=${PY_TOOLS_DIR}:$PATH python -m pip uninstall pip -y
    PATH=${PY_TOOLS_DIR}:$PATH python -m ensurepip

    pushd ${PYENV_ROOT}/versions/${version_alias}/tools/Scripts/
      ln -s pip3.exe pip.exe
    popd
  else
    # If there's already a matching directory, we should re-use it
    # otherwise, pyenv install will force-rebuild
    ls -hal "${PYENV_ROOT}/versions/${version_alias}/" || true
    if [ ! -d "${PYENV_ROOT}/versions/${version_alias}/" ]; then
      VERSION_ALIAS=${version_alias} MAKEOPTS=-j${DS_CPU_COUNT} pyenv install ${version}
    fi;
  fi
}

maybe_numpy_min_version()
{
    local pyver=$1

    unset NUMPY_BUILD_VERSION
    unset NUMPY_DEP_VERSION

    # We set >= and < to make sure we have no numpy incompatibilities
    # otherwise, `from deepspeech.impl` throws with "illegal instruction"

    ARCH=$(uname -m)
    case "${OS}:${ARCH}" in
        Linux:x86_64)
            case "${pyver}" in
                3.7*)
                    export NUMPY_BUILD_VERSION="==1.14.5"
                    export NUMPY_DEP_VERSION=">=1.14.5"
                ;;
                3.8*)
                    export NUMPY_BUILD_VERSION="==1.17.3"
                    export NUMPY_DEP_VERSION=">=1.17.3"
                ;;
            esac
        ;;

        Darwin:*)
            case "${pyver}" in
                3.5*|3.6)
                    export NUMPY_BUILD_VERSION="==1.9.0"
                    export NUMPY_DEP_VERSION=">=1.9.0"
                ;;
                3.7*)
                    export NUMPY_BUILD_VERSION="==1.14.5"
                    export NUMPY_DEP_VERSION=">=1.14.5,<=1.17.0"
                ;;
                3.8*)
                    export NUMPY_BUILD_VERSION="==1.17.3"
                    export NUMPY_DEP_VERSION=">=1.17.3,<=1.17.3"
                ;;
            esac
        ;;

        ${TC_MSYS_VERSION}:x86_64)
            case "${pyver}" in
                3.5*)
                    export NUMPY_BUILD_VERSION="==1.11.0"
                    export NUMPY_DEP_VERSION=">=1.11.0,<1.12.0"
                ;;
                3.6*)
                    export NUMPY_BUILD_VERSION="==1.12.0"
                    export NUMPY_DEP_VERSION=">=1.12.0,<1.14.5"
                ;;
                3.7*)
                    export NUMPY_BUILD_VERSION="==1.14.5"
                    export NUMPY_DEP_VERSION=">=1.14.5,<=1.17.0"
                ;;
                3.8*)
                    export NUMPY_BUILD_VERSION="==1.17.3"
                    export NUMPY_DEP_VERSION=">=1.17.3,<=1.17.3"
                ;;
            esac
        ;;

        *)
            export NUMPY_BUILD_VERSION="==1.7.0"
            export NUMPY_DEP_VERSION=">=1.7.0"
        ;;
    esac
}

get_python_pkg_url()
{
  local pyver_pkg=$1
  local py_unicode_type=$2

  local pkgname=$3
  if [ -z "${pkgname}" ]; then
    pkgname="deepspeech"
  fi

  local root=$4
  if [ -z "${root}" ]; then
    root="${DEEPSPEECH_ARTIFACTS_ROOT}"
  fi

  local platform=$(python -c 'import sys; import platform; plat = platform.system().lower(); arch = platform.machine().lower(); plat = "manylinux1" if plat == "linux" and arch == "x86_64" else plat; plat = "macosx_10_10" if plat == "darwin" else plat; plat = "win" if plat == "windows" else plat; sys.stdout.write("%s_%s" % (plat, platform.machine().lower()));')
  local whl_ds_version="$(python -c 'from pkg_resources import parse_version; print(parse_version("'${DS_VERSION}'"))')"
  local deepspeech_pkg="${pkgname}-${whl_ds_version}-cp${pyver_pkg}-cp${pyver_pkg}${py_unicode_type}-${platform}.whl"

  echo "${root}/${deepspeech_pkg}"
}

get_tflite_python_pkg_name()
{
  # Default to deepspeech package
  local _pkgname="deepspeech_tflite"

  ARCH=$(uname -m)
  case "${OS}:${ARCH}" in
      Linux:armv7l|Linux:aarch64)
          # On linux/arm or linux/aarch64 we don't produce deepspeech_tflite
          _pkgname="deepspeech"
      ;;

      *)
          _pkgname="deepspeech_tflite"
      ;;
  esac

  echo "${_pkgname}"
}

extract_python_versions()
{
  # call extract_python_versions ${pyver_full} pyver pyver_pkg py_unicode_type pyconf pyalias
  local _pyver_full=$1

  if [ -z "${_pyver_full}" ]; then
      echo "No python version given, aborting."
      exit 1
  fi;

  local _pyver=$(echo "${_pyver_full}" | cut -d':' -f1)

  # 3.8.x => 38
  local _pyver_pkg=$(echo "${_pyver}" | cut -d'.' -f1,2 | tr -d '.')

  # https://www.python.org/dev/peps/pep-3149/#proposal
  # 'm' => pymalloc
  # 'u' => wide unicode
  local _py_unicode_type=$(echo "${_pyver_full}" | cut -d':' -f2)
  if [ "${_py_unicode_type}" = "m" ]; then
    local _pyconf="ucs2"
  elif [ "${_py_unicode_type}" = "mu" ]; then
    local _pyconf="ucs4"
  elif [ "${_py_unicode_type}" = "" ]; then # valid for Python 3.8
    local _pyconf="ucs2"
  fi;

  local _pyalias="${_pyver}_${_pyconf}"

  eval "${2}=${_pyver}"
  eval "${3}=${_pyver_pkg}"
  eval "${4}=${_py_unicode_type}"
  eval "${5}=${_pyconf}"
  eval "${6}=${_pyalias}"
}
