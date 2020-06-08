# Need devel version cause we need /usr/include/cudnn.h 
# for compiling libctc_decoder_with_kenlm.so
FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04


# >> START Install base software

# Get basic packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        apt-utils \
        build-essential \
        curl \
        wget \
        git \
        python3 \
        python3-dev \
        python3-pip \
        python3-wheel \
        python3-numpy \
        libcurl3-dev  \
        ca-certificates \
        gcc \
        sox \
        libsox-fmt-mp3 \
        htop \
        nano \
        cmake \
        libboost-all-dev \
        zlib1g-dev \
        libbz2-dev \
        liblzma-dev \
        locales \
        pkg-config \
        libpng-dev \
        libsox-dev \
        libmagic-dev \
        libgsm1-dev \
        libltdl-dev \
        openjdk-8-jdk \
        bash-completion \
        g++ \
        unzip

RUN ln -s -f /usr/bin/python3 /usr/bin/python

# Install NCCL 2.2
RUN apt-get --no-install-recommends install -qq -y --allow-downgrades --allow-change-held-packages libnccl2=2.3.7-1+cuda10.0 libnccl-dev=2.3.7-1+cuda10.0

# Install Bazel
RUN curl -LO "https://github.com/bazelbuild/bazel/releases/download/0.24.1/bazel_0.24.1-linux-x86_64.deb"
RUN dpkg -i bazel_*.deb

# Install CUDA CLI Tools
RUN apt-get --no-install-recommends install -qq -y cuda-command-line-tools-10-0

# Install pip
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py

# << END Install base software




# >> START Configure Tensorflow Build

# Clone TensorFlow from Mozilla repo
RUN git clone https://github.com/mozilla/tensorflow/
WORKDIR /tensorflow
RUN git checkout r1.15


# GPU Environment Setup
ENV TF_NEED_CUDA 1
ENV TF_CUDA_PATHS "/usr/local/cuda,/usr/lib/x86_64-linux-gnu/"
ENV TF_CUDA_VERSION 10.0
ENV TF_CUDNN_VERSION 7
ENV TF_CUDA_COMPUTE_CAPABILITIES 6.0
ENV TF_NCCL_VERSION 2.3

# Common Environment Setup
ENV TF_BUILD_CONTAINER_TYPE GPU
ENV TF_BUILD_OPTIONS OPT
ENV TF_BUILD_DISABLE_GCP 1
ENV TF_BUILD_ENABLE_XLA 0
ENV TF_BUILD_PYTHON_VERSION PYTHON3
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
ENV PYTHON_BIN_PATH /usr/bin/python3.6
ENV PYTHON_LIB_PATH /usr/lib/python3.6/dist-packages

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

# Alternative clone from GitHub 
# RUN apt-get update && apt-get install -y git-lfs 
# WORKDIR /
# RUN git lfs install
# RUN git clone https://github.com/mozilla/DeepSpeech.git

WORKDIR /DeepSpeech

RUN DS_NODECODER=1 pip3 --no-cache-dir install .

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
RUN bazel build --workspace_status_command="bash native_client/bazel_workspace_status_cmd.sh" --config=monolithic --config=cuda -c opt --copt=-O3 --copt="-D_GLIBCXX_USE_CXX11_ABI=0" --copt=-mtune=generic --copt=-march=x86-64 --copt=-msse --copt=-msse2 --copt=-msse3 --copt=-msse4.1 --copt=-msse4.2 --copt=-mavx --copt=-fvisibility=hidden //native_client:libdeepspeech.so --verbose_failures --action_env=LD_LIBRARY_PATH=${LD_LIBRARY_PATH}

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
# RUN pip3 install /tmp/tensorflow_pkg/*.whl

# Copy built libs to /DeepSpeech/native_client
RUN cp /tensorflow/bazel-bin/native_client/libdeepspeech.so /DeepSpeech/native_client/

# Install TensorFlow
WORKDIR /DeepSpeech/
RUN pip3 install tensorflow-gpu==1.15.0


# Build client.cc and install Python client and decoder bindings
ENV TFDIR /tensorflow
WORKDIR /DeepSpeech/native_client
RUN make deepspeech

WORKDIR /DeepSpeech
RUN cd native_client/python && make bindings
RUN pip3 install --upgrade native_client/python/dist/*.whl

RUN cd native_client/ctcdecode && make bindings
RUN pip3 install --upgrade native_client/ctcdecode/dist/*.whl


# << END Build and bind




# Allow Python printing utf-8
ENV PYTHONIOENCODING UTF-8

# Build KenLM in /DeepSpeech/native_client/kenlm folder
WORKDIR /DeepSpeech/native_client
RUN rm -rf kenlm \
    && git clone --depth 1 https://github.com/kpu/kenlm && cd kenlm \
    && mkdir -p build \
    && cd build \
    && cmake .. \
    && make -j 4

# Done
WORKDIR /DeepSpeech
