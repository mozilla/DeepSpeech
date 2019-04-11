#!/bin/bash
#
# USAGE: ./build-gen-trie.sh DEEPSPEECH_DIR OUTPUT_DIR
# e.g.:  ./build-gen-trie.sh /home/user/DeepSpeech /home/user/Desktop
#
# Info: This script will build generate_trie and save it to the specified output
#       directory. There are some assumptions made on versions of Bazel and
#       TensorFlow currently.
#

set -xe

DEEPSPEECH_DIR=$1
OUTPUT_DIR=$2

if [ ! -d $DEEPSPEECH_DIR ]; then
    echo "$0: ERROR: DEEPSPEECH_DIR does not exist"
    exit
elif [ ! -d $OUTPUT_DIR ]; then
    echo "$0: ERROR: OUTPUT_DIR does not exist"
    exit
fi

echo "Installing dependencies"
sudo apt-get install -y build-essential cmake libboost-all-dev zlib1g-dev libbz2-dev liblzma-dev openjdk-8-jdk bash-completion unzip

mkdir -p /tmp/tf
pushd /tmp/tf
  echo "Downloading and Installing Bazel to tmp dir"
  curl -LO "https://github.com/bazelbuild/bazel/releases/download/0.19.2/bazel_0.19.2-linux-x86_64.deb"
  sudo dpkg -i bazel_*.deb

  echo "Downloading and Installing Mozilla's r1.13 fork of TensorFlow to tmp dir"
  git clone --depth 1 --branch r1.13 http://github.com/mozilla/tensorflow

  pushd tensorflow
    # softlink native_client dir to tensorflow
    ln -s ${DEEPSPEECH_DIR}/native_client .
    # Configure and build generate_trie
    export TF_NEED_CUDA=0
    export TF_ENABLE_XLA=0
    export TF_NEED_JEMALLOC=1
    export TF_NEED_OPENCL_SYCL=0
    export TF_NEED_MKL=0
    export TF_NEED_VERBS=0
    export TF_NEED_MPI=0
    export TF_NEED_IGNITE=0
    export TF_NEED_GDR=0
    export TF_NEED_NGRAPH=0
    export TF_DOWNLOAD_CLANG=0
    export TF_SET_ANDROID_WORKSPACE=0
    export TF_NEED_TENSORRT=0
    export GCC_HOST_COMPILER_PATH=/usr/bin/gcc
    ./configure
    echo "Building generate_trie with Bazel"
    bazel build --config=monolithic -c opt --copt=-march=native --copt=-fvisibility=hidden //native_client:generate_trie
  popd # tensorflow

  generate_trie_binary=/tmp/tf/tensorflow/bazel-bin/native_client/generate_trie
  cp ${generate_trie_binary} ${OUTPUT_DIR}/generate_trie
popd # /tmp/tf

echo "$0: FINISHED Building generate_trie"
echo "$0: Please find your compiled generate_trie binary here: ${OUTPUT_DIR}/generate_trie"
