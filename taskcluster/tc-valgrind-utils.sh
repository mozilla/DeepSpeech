#!/bin/bash

set -xe

run_valgrind_basic()
{
    valgrind \
        --leak-check=full \
	--leak-resolution=high \
	--show-reachable=yes \
	--track-origins=yes \
	--log-file=${TASKCLUSTER_ARTIFACTS}/valgrind_basic.log \
	deepspeech \
            --model ${TASKCLUSTER_TMP_DIR}/${model_name_mmap} \
	    --scorer ${TASKCLUSTER_TMP_DIR}/kenlm.scorer \
	    --audio ${TASKCLUSTER_TMP_DIR}/${ldc93s1_sample_filename}
	    --stream 320 \
	    -t
}

run_valgrind_stream()
{
    valgrind \
        --leak-check=full \
	--leak-resolution=high \
	--show-reachable=yes \
	--track-origins=yes \
	--log-file=${TASKCLUSTER_ARTIFACTS}/valgrind_stream.log \
	deepspeech \
            --model ${TASKCLUSTER_TMP_DIR}/${model_name_mmap} \
	    --scorer ${TASKCLUSTER_TMP_DIR}/kenlm.scorer \
	    --audio ${TASKCLUSTER_TMP_DIR}/${ldc93s1_sample_filename}
	    --stream 320 -t
}
