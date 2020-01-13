#!/bin/bash

set -xe

source $(dirname "$0")/tc-tests-utils.sh

model_source=${DEEPSPEECH_TEST_MODEL//.pb/.tflite}
model_name=$(basename "${model_source}")

download_benchmark_model "${TASKCLUSTER_TMP_DIR}/ds"

export PATH=${TASKCLUSTER_TMP_DIR}/ds/:$PATH

lite_benchmark_model \
	--graph=${TASKCLUSTER_TMP_DIR}/ds/${model_name} \
	--show_flops \
	--input_layer=input_node,previous_state_c,previous_state_h,input_samples \
	--input_layer_type=float,float,float,float \
	--input_layer_shape=1,16,19,26:1,100:1,100:512 \
	--output_layer=logits,new_state_c,new_state_h,mfccs
