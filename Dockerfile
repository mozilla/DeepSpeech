# Need devel version cause we need /usr/include/cudnn.h 
# for compiling libctc_decoder_with_kenlm.so
FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04


# >> START Install base software

# Get basic packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        wget \
        git \
        python \
        python-dev \
        python-pip \
        python-wheel \
        python-numpy \
        libcurl3-dev  \
        ca-certificates \
        gcc \
        sox \
        libsox-fmt-mp3 \
        htop \
        nano \
        swig \
        cmake \
        libboost-all-dev \
        zlib1g-dev \
        libbz2-dev \
        liblzma-dev \
        locales \
        pkg-config \
        libsox-dev

# Install NCCL 2.2
RUN apt-get install -qq -y --allow-downgrades --allow-change-held-packages libnccl2=2.2.13-1+cuda9.0 libnccl-dev=2.2.13-1+cuda9.0


# Install Bazel
RUN apt-get install -y openjdk-8-jdk
# Use bazel 0.11.1 cause newer bazel fails to compile TensorFlow (https://github.com/tensorflow/tensorflow/issues/18450#issuecomment-381380000)
RUN apt-get install -y --no-install-recommends bash-completion g++ zlib1g-dev
RUN curl -LO "https://github.com/bazelbuild/bazel/releases/download/0.15.2/bazel_0.15.2-linux-x86_64.deb"
RUN dpkg -i bazel_*.deb

# Install CUDA CLI Tools
RUN apt-get install -qq -y cuda-command-line-tools-9-0

# Install pip
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

# << END Install base software




# >> START Configure Tensorflow Build

# Clone TensoFlow from Mozilla repo
RUN git clone https://github.com/mozilla/tensorflow/
WORKDIR /tensorflow
RUN git checkout r1.12


# GPU Environment Setup
ENV TF_NEED_CUDA 1
ENV CUDA_TOOLKIT_PATH /usr/local/cuda
ENV CUDA_PKG_VERSION 9-0=9.0.176-1
ENV CUDA_VERSION 9.0.176
ENV TF_CUDA_VERSION 9.0
ENV TF_CUDNN_VERSION 7.3.0
ENV CUDNN_INSTALL_PATH /usr/lib/x86_64-linux-gnu/
ENV TF_CUDA_COMPUTE_CAPABILITIES 6.0
ENV TF_NCCL_VERSION 2.2.13
# ENV NCCL_INSTALL_PATH /usr/lib/x86_64-linux-gnu/

# Common Environment Setup
ENV TF_BUILD_CONTAINER_TYPE GPU
ENV TF_BUILD_OPTIONS OPT
ENV TF_BUILD_DISABLE_GCP 1
ENV TF_BUILD_ENABLE_XLA 0
ENV TF_BUILD_PYTHON_VERSION PYTHON2
ENV TF_BUILD_IS_OPT OPT
ENV TF_BUILD_IS_PIP PIP

# Other Parameters
ENV CC_OPT_FLAGS -mavx -mavx2 -msse4.1 -msse4.2 -mfma
ENV TF_NEED_GCP 0
ENV TF_NEED_HDFS 0
ENV TF_NEED_JEMALLOC 1
ENV TF_NEED_OPENCL 0
ENV TF_CUDA_CLANG 0
ENV TF_NEED_MKL 0
ENV TF_ENABLE_XLA 0
ENV TF_NEED_AWS 0
ENV TF_NEED_KAFKA 0
ENV TF_NEED_NGRAPH 0
ENV TF_DOWNLOAD_CLANG 0
ENV TF_NEED_TENSORRT 0
ENV TF_NEED_GDR 0
ENV TF_NEED_VERBS 0
ENV TF_NEED_OPENCL_SYCL 0
ENV PYTHON_BIN_PATH /usr/bin/python2.7
ENV PYTHON_LIB_PATH /usr/lib/python2.7/dist-packages

# << END Configure Tensorflow Build




# >> START Configure Bazel

# Running bazel inside a `docker build` command causes trouble, cf:
#   https://github.com/bazelbuild/bazel/issues/134
# The easiest solution is to set up a bazelrc file forcing --batch.
RUN echo "startup --batch" >>/etc/bazel.bazelrc
# Similarly, we need to workaround sandboxing issues:
#   https://github.com/bazelbuild/bazel/issues/418
RUN echo "build --spawn_strategy=standalone --genrule_strategy=standalone" \
    >>/etc/bazel.bazelrc

# Put cuda libraries to where they are expected to be
RUN mkdir /usr/local/cuda/lib &&  \
    ln -s /usr/lib/x86_64-linux-gnu/libnccl.so.2 /usr/local/cuda/lib/libnccl.so.2 && \
    ln -s /usr/include/nccl.h /usr/local/cuda/include/nccl.h && \
    ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 && \
    ln -s /usr/include/cudnn.h /usr/local/cuda/include/cudnn.h


# Set library paths
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu/:/usr/local/cuda/lib64/stubs/

# << END Configure Bazel




# Copy DeepSpeech repo contents to container's /DeepSpeech
COPY . /DeepSpeech/

WORKDIR /DeepSpeech

RUN pip --no-cache-dir install -r requirements.txt

# Link DeepSpeech native_client libs to tf folder
RUN ln -s /DeepSpeech/native_client /tensorflow




# >> START Build and bind

WORKDIR /tensorflow

# Fix for not found script https://github.com/tensorflow/tensorflow/issues/471
RUN ./configure

# Using CPU optimizations:
# -mtune=generic -march=x86-64 -msse -msse2 -msse3 -msse4.1 -msse4.2 -mavx.
# Adding --config=cuda flag to build using CUDA.

# passing LD_LIBRARY_PATH is required cause Bazel doesn't pickup it from environment


# Build DeepSpeech
RUN bazel build --config=monolithic --config=cuda -c opt --copt=-O3 --copt="-D_GLIBCXX_USE_CXX11_ABI=0" --copt=-mtune=generic --copt=-march=x86-64 --copt=-msse --copt=-msse2 --copt=-msse3 --copt=-msse4.1 --copt=-msse4.2 --copt=-mavx --copt=-fvisibility=hidden //native_client:libdeepspeech.so //native_client:generate_trie --verbose_failures --action_env=LD_LIBRARY_PATH=${LD_LIBRARY_PATH}

###
### Using TensorFlow upstream should work
###
# # Build TF pip package
# RUN bazel build --config=opt --config=cuda --copt="-D_GLIBCXX_USE_CXX11_ABI=0" --copt=-mtune=generic --copt=-march=x86-64 --copt=-msse --copt=-msse2 --copt=-msse3 --copt=-msse4.1 --copt=-msse4.2 --copt=-mavx //tensorflow/tools/pip_package:build_pip_package --verbose_failures --action_env=LD_LIBRARY_PATH=${LD_LIBRARY_PATH}
#
# # Build wheel
# RUN bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
#
# # Install tensorflow from our custom wheel
# RUN pip install /tmp/tensorflow_pkg/*.whl

# Copy built libs to /DeepSpeech/native_client
RUN cp /tensorflow/bazel-bin/native_client/generate_trie /DeepSpeech/native_client/ \
    && cp /tensorflow/bazel-bin/native_client/libdeepspeech.so /DeepSpeech/native_client/

# Install TensorFlow
WORKDIR /DeepSpeech/
RUN pip install tensorflow-gpu==1.12.0rc2


# Make DeepSpeech and install Python bindings
ENV TFDIR /tensorflow
WORKDIR /DeepSpeech/native_client
RUN make deepspeech
WORKDIR /DeepSpeech/native_client/python
RUN make bindings
RUN pip install dist/deepspeech*
WORKDIR /DeepSpeech/native_client/ctcdecode
RUN make
RUN pip install dist/*.whl


# << END Build and bind




# Allow Python printing utf-8
ENV PYTHONIOENCODING UTF-8

# Build KenLM in /DeepSpeech/native_client/kenlm folder
WORKDIR /DeepSpeech/native_client
RUN rm -rf kenlm \
    && git clone https://github.com/kpu/kenlm && cd kenlm \    
    && mkdir -p build \
    && cd build \
    && cmake .. \
    && make -j 4

# Done
WORKDIR /DeepSpeech
