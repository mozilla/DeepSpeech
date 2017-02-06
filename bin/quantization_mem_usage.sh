#/bin/sh

TFDIR=${TFDIR:-../tensorflow}
MODEL=${MODEL:-test.frozen.ldc93s1.pb}
m=$(basename "${MODEL}")

WAV=./data/ldc93s1/LDC93S1.wav
export LD_LIBRARY_PATH=${TFDIR}/bazel-bin/tensorflow:${TFDIR}/bazel-bin/native_client:${LD_LIBRARY_PATH}

valgrind --tool=massif --alloc-fn='Eigen::internal::aligned_malloc(unsigned long)' --massif-out-file="massif.deepspeech.${m}.%p" ./native_client/deepspeech "${MODEL}" "${WAV}"
