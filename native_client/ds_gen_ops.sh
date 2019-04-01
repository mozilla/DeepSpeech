#!/bin/sh

if [ "${TF_NEED_CUDA}" -eq "1" ]; then
    ops_to_register="ops_to_register_gpu.h"
else
    ops_to_register="ops_to_register_cpu.h"
fi;

TF_DIR="$(realpath $(pwd)/tensorflow/../)"
if [ ! -d "${TF_DIR}" ]; then
   return 1
fi;

cp "native_client/${ops_to_register}" "${TF_DIR}/tensorflow/core/framework/ops_to_register.h"

cat "native_client/${ops_to_register}"
