#!/bin/bash

set -xe

arm_flavor=$1
api_level=$2

source $(dirname "$0")/tc-tests-utils.sh

# Android test code only supports 16kHz and hard-coded filename
ldc93s1_sample_filename='LDC93S1.wav'

model_source=${DEEPSPEECH_TEST_MODEL//.pb/.tflite}
model_name=$(basename "${model_source}")

download_data

force_java_apk_x86_64

android_start_emulator "${arm_flavor}" "${api_level}"

# Required, because of "gradle connectedAndroidTest" deps
do_deepspeech_java_apk_build

# We need to wait for emulator to be running, at least package service
android_wait_for_emulator

android_setup_apk_data

android_run_tests

android_stop_emulator
