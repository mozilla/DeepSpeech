#!/bin/bash

set -xe

strip() {
  echo "$(echo $1 | sed -e 's/^[[:space:]]+//' -e 's/[[:space:]]+$//')"
}

# This verify exact inference result
assert_correct_inference()
{
  phrase=$(strip "$1")
  expected=$(strip "$2")
  status=$3

  if [ "$status" -ne "0" ]; then
      case "$(cat ${TASKCLUSTER_TMP_DIR}/stderr)" in
          *"incompatible with minimum version"*)
              echo "Prod model too old for client, skipping test."
              return 0
          ;;

          *)
              echo "Client failed to run:"
              cat ${TASKCLUSTER_TMP_DIR}/stderr
              return 1
          ;;
      esac
  fi

  if [ -z "${phrase}" -o -z "${expected}" ]; then
      echo "One or more empty strings:"
      echo "phrase: <${phrase}>"
      echo "expected: <${expected}>"
      return 1
  fi;

  if [ "${phrase}" = "${expected}" ]; then
      echo "Proper output has been produced:"
      echo "${phrase}"
      return 0
  else
      echo "!! Non matching output !!"
      echo "got: <${phrase}>"
      if [ -x "$(command -v xxd)" ]; then
        echo "xxd:"; echo "${phrase}" | xxd
      fi
      echo "-------------------"
      echo "expected: <${expected}>"
      if [ -x "$(command -v xxd)" ]; then
        echo "xxd:"; echo "${expected}" | xxd
      fi
      return 1
  fi;
}

# This verify that ${expected} is contained within ${phrase}
assert_working_inference()
{
  phrase=$1
  expected=$2
  status=$3

  if [ -z "${phrase}" -o -z "${expected}" ]; then
      echo "One or more empty strings:"
      echo "phrase: <${phrase}>"
      echo "expected: <${expected}>"
      return 1
  fi;

  if [ "$status" -ne "0" ]; then
      case "$(cat ${TASKCLUSTER_TMP_DIR}/stderr)" in
          *"incompatible with minimum version"*)
              echo "Prod model too old for client, skipping test."
              return 0
          ;;

          *)
              echo "Client failed to run:"
              cat ${TASKCLUSTER_TMP_DIR}/stderr
              return 1
          ;;
      esac
  fi

  case "${phrase}" in
      *${expected}*)
          echo "Proper output has been produced:"
          echo "${phrase}"
          return 0
      ;;

      *)
          echo "!! Non matching output !!"
          echo "got: <${phrase}>"
          if [ -x "$(command -v xxd)" ]; then
            echo "xxd:"; echo "${phrase}" | xxd
          fi
          echo "-------------------"
          echo "expected: <${expected}>"
          if [ -x "$(command -v xxd)" ]; then
            echo "xxd:"; echo "${expected}" | xxd
          fi
          return 1
      ;;
  esac
}

assert_shows_something()
{
  stderr=$1
  expected=$2

  if [ -z "${stderr}" -o -z "${expected}" ]; then
      echo "One or more empty strings:"
      echo "stderr: <${stderr}>"
      echo "expected: <${expected}>"
      return 1
  fi;

  case "${stderr}" in
      *"incompatible with minimum version"*)
          echo "Prod model too old for client, skipping test."
          return 0
      ;;

      *${expected}*)
          echo "Proper output has been produced:"
          echo "${stderr}"
          return 0
      ;;

      *)
          echo "!! Non matching output !!"
          echo "got: <${stderr}>"
          if [ -x "$(command -v xxd)" ]; then
            echo "xxd:"; echo "${stderr}" | xxd
          fi
          echo "-------------------"
          echo "expected: <${expected}>"
          if [ -x "$(command -v xxd)" ]; then
            echo "xxd:"; echo "${expected}" | xxd
          fi
          return 1
      ;;
  esac
}

assert_not_present()
{
  stderr=$1
  not_expected=$2

  if [ -z "${stderr}" -o -z "${not_expected}" ]; then
      echo "One or more empty strings:"
      echo "stderr: <${stderr}>"
      echo "not_expected: <${not_expected}>"
      return 1
  fi;

  case "${stderr}" in
      *${not_expected}*)
          echo "!! Not expected was present !!"
          echo "got: <${stderr}>"
          if [ -x "$(command -v xxd)" ]; then
            echo "xxd:"; echo "${stderr}" | xxd
          fi
          echo "-------------------"
          echo "not_expected: <${not_expected}>"
          if [ -x "$(command -v xxd)" ]; then
            echo "xxd:"; echo "${not_expected}" | xxd
          fi
          return 1
      ;;

      *)
          echo "Proper not expected output has not been produced:"
          echo "${stderr}"
          return 0
      ;;
  esac
}

assert_correct_ldc93s1()
{
  assert_correct_inference "$1" "she had your dark suit in greasy wash water all year" "$2"
}

assert_working_ldc93s1()
{
  assert_working_inference "$1" "she had your dark suit in greasy wash water all year" "$2"
}

assert_correct_ldc93s1_lm()
{
  assert_correct_inference "$1" "she had your dark suit in greasy wash water all year" "$2"
}

assert_working_ldc93s1_lm()
{
  assert_working_inference "$1" "she had your dark suit in greasy wash water all year" "$2"
}

assert_correct_multi_ldc93s1()
{
  assert_shows_something "$1" "/${ldc93s1_sample_filename}%she had your dark suit in greasy wash water all year%" "$?"
  assert_shows_something "$1" "/LDC93S1_pcms16le_2_44100.wav%she had your dark suit in greasy wash water all year%" "$?"
  ## 8k will output garbage anyway ...
  # assert_shows_something "$1" "/LDC93S1_pcms16le_1_8000.wav%she hayorasryrtl lyreasy asr watal w water all year%"
}

assert_correct_ldc93s1_prodmodel()
{
  if [ -z "$3" -o "$3" = "16k" ]; then
    assert_correct_inference "$1" "she had your dark suit in greasy wash water all year" "$2"
  fi;

  if [ "$3" = "8k" ]; then
    assert_correct_inference "$1" "she had to do suit in greasy wash water all year" "$2"
  fi;
}

assert_correct_ldc93s1_prodtflitemodel()
{
  if [ -z "$3" -o "$3" = "16k" ]; then
    assert_correct_inference "$1" "she had her dark suit in greasy wash water all year" "$2"
  fi;

  if [ "$3" = "8k" ]; then
    assert_correct_inference "$1" "she had to do so in greasy wash water all year" "$2"
  fi;
}

assert_correct_ldc93s1_prodmodel_stereo_44k()
{
  assert_correct_inference "$1" "she had your dark suit in greasy wash water all year" "$2"
}

assert_correct_ldc93s1_prodtflitemodel_stereo_44k()
{
  assert_correct_inference "$1" "she had her dark suit in greasy wash water all year" "$2"
}

assert_correct_warning_upsampling()
{
  assert_shows_something "$1" "erratic speech recognition"
}

assert_tensorflow_version()
{
  assert_shows_something "$1" "${EXPECTED_TENSORFLOW_VERSION}"
}

assert_deepspeech_version()
{
  assert_not_present "$1" "DeepSpeech: unknown"
}

# We need to ensure that running on inference really leverages GPU because
# it might default back to CPU
ensure_cuda_usage()
{
  local _maybe_cuda=$1
  DS_BINARY_FILE=${DS_BINARY_FILE:-"deepspeech"}

  if [ "${_maybe_cuda}" = "cuda" ]; then
    set +e
    export TF_CPP_MIN_VLOG_LEVEL=1
    ds_cuda=$(${DS_BINARY_PREFIX}${DS_BINARY_FILE} --model ${TASKCLUSTER_TMP_DIR}/${model_name} --audio ${TASKCLUSTER_TMP_DIR}/${ldc93s1_sample_filename} 2>&1 1>/dev/null)
    export TF_CPP_MIN_VLOG_LEVEL=
    set -e

    assert_shows_something "${ds_cuda}" "Successfully opened dynamic library nvcuda.dll"
    assert_not_present "${ds_cuda}" "Skipping registering GPU devices"
  fi;
}

check_versions()
{
  set +e
  ds_help=$(${DS_BINARY_PREFIX}deepspeech --model ${TASKCLUSTER_TMP_DIR}/${model_name} --audio ${TASKCLUSTER_TMP_DIR}/${ldc93s1_sample_filename} 2>&1 1>/dev/null)
  set -e

  assert_tensorflow_version "${ds_help}"
  assert_deepspeech_version "${ds_help}"
}

assert_deepspeech_runtime()
{
  local expected_runtime=$1

  set +e
  local ds_version=$(${DS_BINARY_PREFIX}deepspeech --version 2>&1)
  set -e

  assert_shows_something "${ds_version}" "${expected_runtime}"
}

check_runtime_nodejs()
{
  assert_deepspeech_runtime "Runtime: Node"
}

check_runtime_electronjs()
{
  assert_deepspeech_runtime "Runtime: Electron"
}

run_tflite_basic_inference_tests()
{
  set +e
  phrase_pbmodel_nolm=$(${DS_BINARY_PREFIX}deepspeech --model ${DATA_TMP_DIR}/${model_name} --audio ${DATA_TMP_DIR}/${ldc93s1_sample_filename} 2>${TASKCLUSTER_TMP_DIR}/stderr)
  set -e
  assert_correct_ldc93s1 "${phrase_pbmodel_nolm}" "$?"

  set +e
  phrase_pbmodel_nolm=$(${DS_BINARY_PREFIX}deepspeech --model ${DATA_TMP_DIR}/${model_name} --audio ${DATA_TMP_DIR}/${ldc93s1_sample_filename} --extended 2>${TASKCLUSTER_TMP_DIR}/stderr)
  set -e
  assert_correct_ldc93s1 "${phrase_pbmodel_nolm}" "$?"
}

run_netframework_inference_tests()
{
  set +e
  phrase_pbmodel_nolm=$(DeepSpeechConsole.exe --model ${TASKCLUSTER_TMP_DIR}/${model_name} --audio ${TASKCLUSTER_TMP_DIR}/${ldc93s1_sample_filename} 2>${TASKCLUSTER_TMP_DIR}/stderr)
  set -e
  assert_working_ldc93s1 "${phrase_pbmodel_nolm}" "$?"

  set +e
  phrase_pbmodel_nolm=$(DeepSpeechConsole.exe --model ${TASKCLUSTER_TMP_DIR}/${model_name} --audio ${TASKCLUSTER_TMP_DIR}/${ldc93s1_sample_filename} --extended yes 2>${TASKCLUSTER_TMP_DIR}/stderr)
  set -e
  assert_working_ldc93s1 "${phrase_pbmodel_nolm}" "$?"

  set +e
  phrase_pbmodel_nolm=$(DeepSpeechConsole.exe --model ${TASKCLUSTER_TMP_DIR}/${model_name_mmap} --audio ${TASKCLUSTER_TMP_DIR}/${ldc93s1_sample_filename} 2>${TASKCLUSTER_TMP_DIR}/stderr)
  set -e
  assert_working_ldc93s1 "${phrase_pbmodel_nolm}" "$?"

  set +e
  phrase_pbmodel_withlm=$(DeepSpeechConsole.exe --model ${TASKCLUSTER_TMP_DIR}/${model_name_mmap} --scorer ${TASKCLUSTER_TMP_DIR}/kenlm.scorer --audio ${TASKCLUSTER_TMP_DIR}/${ldc93s1_sample_filename} 2>${TASKCLUSTER_TMP_DIR}/stderr)
  set -e
  assert_working_ldc93s1_lm "${phrase_pbmodel_withlm}" "$?"
}

run_electronjs_inference_tests()
{
  set +e
  phrase_pbmodel_nolm=$(deepspeech --model ${TASKCLUSTER_TMP_DIR}/${model_name} --audio ${TASKCLUSTER_TMP_DIR}/${ldc93s1_sample_filename} 2>${TASKCLUSTER_TMP_DIR}/stderr)
  set -e
  assert_working_ldc93s1 "${phrase_pbmodel_nolm}" "$?"

  set +e
  phrase_pbmodel_nolm=$(deepspeech --model ${TASKCLUSTER_TMP_DIR}/${model_name} --audio ${TASKCLUSTER_TMP_DIR}/${ldc93s1_sample_filename} --extended 2>${TASKCLUSTER_TMP_DIR}/stderr)
  set -e
  assert_working_ldc93s1 "${phrase_pbmodel_nolm}" "$?"

  set +e
  phrase_pbmodel_nolm=$(deepspeech --model ${TASKCLUSTER_TMP_DIR}/${model_name_mmap} --audio ${TASKCLUSTER_TMP_DIR}/${ldc93s1_sample_filename} 2>${TASKCLUSTER_TMP_DIR}/stderr)
  set -e
  assert_working_ldc93s1 "${phrase_pbmodel_nolm}" "$?"

  set +e
  phrase_pbmodel_withlm=$(deepspeech --model ${TASKCLUSTER_TMP_DIR}/${model_name_mmap} --scorer ${TASKCLUSTER_TMP_DIR}/kenlm.scorer --audio ${TASKCLUSTER_TMP_DIR}/${ldc93s1_sample_filename} 2>${TASKCLUSTER_TMP_DIR}/stderr)
  set -e
  assert_working_ldc93s1_lm "${phrase_pbmodel_withlm}" "$?"
}

run_basic_inference_tests()
{
  set +e
  deepspeech --model "" --audio ${TASKCLUSTER_TMP_DIR}/${ldc93s1_sample_filename} 2>${TASKCLUSTER_TMP_DIR}/stderr
  set -e
  grep "Missing model information" ${TASKCLUSTER_TMP_DIR}/stderr

  set +e
  phrase_pbmodel_nolm=$(deepspeech --model ${TASKCLUSTER_TMP_DIR}/${model_name} --audio ${TASKCLUSTER_TMP_DIR}/${ldc93s1_sample_filename} 2>${TASKCLUSTER_TMP_DIR}/stderr)
  status=$?
  set -e
  assert_correct_ldc93s1 "${phrase_pbmodel_nolm}" "$status"

  set +e
  phrase_pbmodel_nolm=$(deepspeech --model ${TASKCLUSTER_TMP_DIR}/${model_name} --audio ${TASKCLUSTER_TMP_DIR}/${ldc93s1_sample_filename} --extended 2>${TASKCLUSTER_TMP_DIR}/stderr)
  status=$?
  set -e
  assert_correct_ldc93s1 "${phrase_pbmodel_nolm}" "$status"

  set +e
  phrase_pbmodel_nolm=$(deepspeech --model ${TASKCLUSTER_TMP_DIR}/${model_name_mmap} --audio ${TASKCLUSTER_TMP_DIR}/${ldc93s1_sample_filename} 2>${TASKCLUSTER_TMP_DIR}/stderr)
  status=$?
  set -e
  assert_correct_ldc93s1 "${phrase_pbmodel_nolm}" "$status"

  set +e
  phrase_pbmodel_withlm=$(deepspeech --model ${TASKCLUSTER_TMP_DIR}/${model_name_mmap} --scorer ${TASKCLUSTER_TMP_DIR}/kenlm.scorer --audio ${TASKCLUSTER_TMP_DIR}/${ldc93s1_sample_filename} 2>${TASKCLUSTER_TMP_DIR}/stderr)
  status=$?
  set -e
  assert_correct_ldc93s1_lm "${phrase_pbmodel_withlm}" "$status"
}

run_all_inference_tests()
{
  run_basic_inference_tests

  set +e
  phrase_pbmodel_nolm_stereo_44k=$(deepspeech --model ${TASKCLUSTER_TMP_DIR}/${model_name_mmap} --audio ${TASKCLUSTER_TMP_DIR}/LDC93S1_pcms16le_2_44100.wav 2>${TASKCLUSTER_TMP_DIR}/stderr)
  status=$?
  set -e
  assert_correct_ldc93s1 "${phrase_pbmodel_nolm_stereo_44k}" "$status"

  set +e
  phrase_pbmodel_withlm_stereo_44k=$(deepspeech --model ${TASKCLUSTER_TMP_DIR}/${model_name_mmap} --scorer ${TASKCLUSTER_TMP_DIR}/kenlm.scorer --audio ${TASKCLUSTER_TMP_DIR}/LDC93S1_pcms16le_2_44100.wav 2>${TASKCLUSTER_TMP_DIR}/stderr)
  status=$?
  set -e
  assert_correct_ldc93s1_lm "${phrase_pbmodel_withlm_stereo_44k}" "$status"

  # Run down-sampling warning test only when we actually perform downsampling
  if [ "${ldc93s1_sample_filename}" != "LDC93S1_pcms16le_1_8000.wav" ]; then
    set +e
    phrase_pbmodel_nolm_mono_8k=$(deepspeech --model ${TASKCLUSTER_TMP_DIR}/${model_name_mmap} --audio ${TASKCLUSTER_TMP_DIR}/LDC93S1_pcms16le_1_8000.wav 2>&1 1>/dev/null)
    set -e
    assert_correct_warning_upsampling "${phrase_pbmodel_nolm_mono_8k}"

    set +e
    phrase_pbmodel_withlm_mono_8k=$(deepspeech --model ${TASKCLUSTER_TMP_DIR}/${model_name_mmap} --scorer ${TASKCLUSTER_TMP_DIR}/kenlm.scorer --audio ${TASKCLUSTER_TMP_DIR}/LDC93S1_pcms16le_1_8000.wav 2>&1 1>/dev/null)
    set -e
    assert_correct_warning_upsampling "${phrase_pbmodel_withlm_mono_8k}"
  fi;
}

run_prod_concurrent_stream_tests()
{
  local _bitrate=$1

  set +e
  output=$(python ${TASKCLUSTER_TMP_DIR}/test_sources/concurrent_streams.py \
             --model ${TASKCLUSTER_TMP_DIR}/${model_name_mmap} \
             --scorer ${TASKCLUSTER_TMP_DIR}/kenlm.scorer \
             --audio1 ${TASKCLUSTER_TMP_DIR}/LDC93S1_pcms16le_1_16000.wav \
             --audio2 ${TASKCLUSTER_TMP_DIR}/new-home-in-the-stars-16k.wav 2>${TASKCLUSTER_TMP_DIR}/stderr)
  status=$?
  set -e

  output1=$(echo "${output}" | head -n 1)
  output2=$(echo "${output}" | tail -n 1)

  assert_correct_ldc93s1_prodmodel "${output1}" "${status}" "16k"
  assert_correct_inference "${output2}" "we must find a new home in the stars" "${status}"
}

run_prod_inference_tests()
{
  local _bitrate=$1

  set +e
  phrase_pbmodel_withlm=$(deepspeech --model ${TASKCLUSTER_TMP_DIR}/${model_name} --scorer ${TASKCLUSTER_TMP_DIR}/kenlm.scorer --audio ${TASKCLUSTER_TMP_DIR}/${ldc93s1_sample_filename} 2>${TASKCLUSTER_TMP_DIR}/stderr)
  status=$?
  set -e
  assert_correct_ldc93s1_prodmodel "${phrase_pbmodel_withlm}" "$status" "${_bitrate}"

  set +e
  phrase_pbmodel_withlm=$(deepspeech --model ${TASKCLUSTER_TMP_DIR}/${model_name_mmap} --scorer ${TASKCLUSTER_TMP_DIR}/kenlm.scorer --audio ${TASKCLUSTER_TMP_DIR}/${ldc93s1_sample_filename} 2>${TASKCLUSTER_TMP_DIR}/stderr)
  status=$?
  set -e
  assert_correct_ldc93s1_prodmodel "${phrase_pbmodel_withlm}" "$status" "${_bitrate}"

  set +e
  phrase_pbmodel_withlm_stereo_44k=$(deepspeech --model ${TASKCLUSTER_TMP_DIR}/${model_name_mmap} --scorer ${TASKCLUSTER_TMP_DIR}/kenlm.scorer --audio ${TASKCLUSTER_TMP_DIR}/LDC93S1_pcms16le_2_44100.wav 2>${TASKCLUSTER_TMP_DIR}/stderr)
  status=$?
  set -e
  assert_correct_ldc93s1_prodmodel_stereo_44k "${phrase_pbmodel_withlm_stereo_44k}" "$status"

  # Run down-sampling warning test only when we actually perform downsampling
  if [ "${ldc93s1_sample_filename}" != "LDC93S1_pcms16le_1_8000.wav" ]; then
    set +e
    phrase_pbmodel_withlm_mono_8k=$(deepspeech --model ${TASKCLUSTER_TMP_DIR}/${model_name_mmap} --scorer ${TASKCLUSTER_TMP_DIR}/kenlm.scorer --audio ${TASKCLUSTER_TMP_DIR}/LDC93S1_pcms16le_1_8000.wav 2>&1 1>/dev/null)
    set -e
    assert_correct_warning_upsampling "${phrase_pbmodel_withlm_mono_8k}"
  fi;
}

run_prodtflite_inference_tests()
{
  local _bitrate=$1

  set +e
  phrase_pbmodel_withlm=$(deepspeech --model ${TASKCLUSTER_TMP_DIR}/${model_name} --scorer ${TASKCLUSTER_TMP_DIR}/kenlm.scorer --audio ${TASKCLUSTER_TMP_DIR}/${ldc93s1_sample_filename} 2>${TASKCLUSTER_TMP_DIR}/stderr)
  status=$?
  set -e
  assert_correct_ldc93s1_prodtflitemodel "${phrase_pbmodel_withlm}" "$status" "${_bitrate}"

  set +e
  phrase_pbmodel_withlm=$(deepspeech --model ${TASKCLUSTER_TMP_DIR}/${model_name_mmap} --scorer ${TASKCLUSTER_TMP_DIR}/kenlm.scorer --audio ${TASKCLUSTER_TMP_DIR}/${ldc93s1_sample_filename} 2>${TASKCLUSTER_TMP_DIR}/stderr)
  status=$?
  set -e
  assert_correct_ldc93s1_prodtflitemodel "${phrase_pbmodel_withlm}" "$status" "${_bitrate}"

  set +e
  phrase_pbmodel_withlm_stereo_44k=$(deepspeech --model ${TASKCLUSTER_TMP_DIR}/${model_name_mmap} --scorer ${TASKCLUSTER_TMP_DIR}/kenlm.scorer --audio ${TASKCLUSTER_TMP_DIR}/LDC93S1_pcms16le_2_44100.wav 2>${TASKCLUSTER_TMP_DIR}/stderr)
  status=$?
  set -e
  assert_correct_ldc93s1_prodtflitemodel_stereo_44k "${phrase_pbmodel_withlm_stereo_44k}" "$status"

  # Run down-sampling warning test only when we actually perform downsampling
  if [ "${ldc93s1_sample_filename}" != "LDC93S1_pcms16le_1_8000.wav" ]; then
    set +e
    phrase_pbmodel_withlm_mono_8k=$(deepspeech --model ${TASKCLUSTER_TMP_DIR}/${model_name_mmap} --scorer ${TASKCLUSTER_TMP_DIR}/kenlm.scorer --audio ${TASKCLUSTER_TMP_DIR}/LDC93S1_pcms16le_1_8000.wav 2>&1 1>/dev/null)
    set -e
    assert_correct_warning_upsampling "${phrase_pbmodel_withlm_mono_8k}"
  fi;
}

run_multi_inference_tests()
{
  set +e -o pipefail
  multi_phrase_pbmodel_nolm=$(deepspeech --model ${TASKCLUSTER_TMP_DIR}/${model_name} --audio ${TASKCLUSTER_TMP_DIR}/ 2>${TASKCLUSTER_TMP_DIR}/stderr | tr '\n' '%')
  status=$?
  set -e +o pipefail
  assert_correct_multi_ldc93s1 "${multi_phrase_pbmodel_nolm}" "$status"

  set +e -o pipefail
  multi_phrase_pbmodel_withlm=$(deepspeech --model ${TASKCLUSTER_TMP_DIR}/${model_name} --scorer ${TASKCLUSTER_TMP_DIR}/kenlm.scorer --audio ${TASKCLUSTER_TMP_DIR}/ 2>${TASKCLUSTER_TMP_DIR}/stderr | tr '\n' '%')
  status=$?
  set -e +o pipefail
  assert_correct_multi_ldc93s1 "${multi_phrase_pbmodel_withlm}" "$status"
}

run_hotword_tests()
{
  set +e
  hotwords_decode=$(${DS_BINARY_PREFIX}deepspeech --model ${TASKCLUSTER_TMP_DIR}/${model_name_mmap} --scorer ${TASKCLUSTER_TMP_DIR}/kenlm.scorer --audio ${TASKCLUSTER_TMP_DIR}/${ldc93s1_sample_filename} --hot_words "foo:0.0,bar:-0.1" 2>${TASKCLUSTER_TMP_DIR}/stderr)
  status=$?
  set -e
  assert_correct_ldc93s1_lm "${hotwords_decode}" "$status"
}

run_android_hotword_tests()
{
  set +e
  hotwords_decode=$(${DS_BINARY_PREFIX}deepspeech --model ${DATA_TMP_DIR}/${model_name} --scorer ${DATA_TMP_DIR}/kenlm.scorer --audio ${DATA_TMP_DIR}/${ldc93s1_sample_filename} --hot_words "foo:0.0,bar:-0.1" 2>${TASKCLUSTER_TMP_DIR}/stderr)
  status=$?
  set -e
  assert_correct_ldc93s1_lm "${hotwords_decode}" "$status"
}

run_cpp_only_inference_tests()
{
  set +e
  phrase_pbmodel_withlm_intermediate_decode=$(deepspeech --model ${TASKCLUSTER_TMP_DIR}/${model_name_mmap} --scorer ${TASKCLUSTER_TMP_DIR}/kenlm.scorer --audio ${TASKCLUSTER_TMP_DIR}/${ldc93s1_sample_filename} --stream 1280 2>${TASKCLUSTER_TMP_DIR}/stderr | tail -n 1)
  status=$?
  set -e
  assert_correct_ldc93s1_lm "${phrase_pbmodel_withlm_intermediate_decode}" "$status"
}

run_js_streaming_inference_tests()
{
  set +e
  phrase_pbmodel_withlm=$(deepspeech --model ${TASKCLUSTER_TMP_DIR}/${model_name_mmap} --scorer ${TASKCLUSTER_TMP_DIR}/kenlm.scorer --audio ${TASKCLUSTER_TMP_DIR}/${ldc93s1_sample_filename} --stream 2>${TASKCLUSTER_TMP_DIR}/stderr | tail -n 1)
  status=$?
  set -e
  assert_correct_ldc93s1_lm "${phrase_pbmodel_withlm}" "$status"

  set +e
  phrase_pbmodel_withlm=$(deepspeech --model ${TASKCLUSTER_TMP_DIR}/${model_name_mmap} --scorer ${TASKCLUSTER_TMP_DIR}/kenlm.scorer --audio ${TASKCLUSTER_TMP_DIR}/${ldc93s1_sample_filename} --stream --extended 2>${TASKCLUSTER_TMP_DIR}/stderr | tail -n 1)
  status=$?
  set -e
  assert_correct_ldc93s1_lm "${phrase_pbmodel_withlm}" "$status"
}

run_js_streaming_prod_inference_tests()
{
  local _bitrate=$1
  set +e
  phrase_pbmodel_withlm=$(deepspeech --model ${TASKCLUSTER_TMP_DIR}/${model_name_mmap} --scorer ${TASKCLUSTER_TMP_DIR}/kenlm.scorer --audio ${TASKCLUSTER_TMP_DIR}/${ldc93s1_sample_filename} --stream 2>${TASKCLUSTER_TMP_DIR}/stderr | tail -n 1)
  status=$?
  set -e
  assert_correct_ldc93s1_prodmodel "${phrase_pbmodel_withlm}" "$status" "${_bitrate}"

  local _bitrate=$1
  set +e
  phrase_pbmodel_withlm=$(deepspeech --model ${TASKCLUSTER_TMP_DIR}/${model_name_mmap} --scorer ${TASKCLUSTER_TMP_DIR}/kenlm.scorer --audio ${TASKCLUSTER_TMP_DIR}/${ldc93s1_sample_filename} --stream --extended 2>${TASKCLUSTER_TMP_DIR}/stderr | tail -n 1)
  status=$?
  set -e
  assert_correct_ldc93s1_prodmodel "${phrase_pbmodel_withlm}" "$status" "${_bitrate}"
}

run_js_streaming_prodtflite_inference_tests()
{
  local _bitrate=$1
  set +e
  phrase_pbmodel_withlm=$(deepspeech --model ${TASKCLUSTER_TMP_DIR}/${model_name_mmap} --scorer ${TASKCLUSTER_TMP_DIR}/kenlm.scorer --audio ${TASKCLUSTER_TMP_DIR}/${ldc93s1_sample_filename} --stream 2>${TASKCLUSTER_TMP_DIR}/stderr | tail -n 1)
  status=$?
  set -e
  assert_correct_ldc93s1_prodtflitemodel "${phrase_pbmodel_withlm}" "$status" "${_bitrate}"

  local _bitrate=$1
  set +e
  phrase_pbmodel_withlm=$(deepspeech --model ${TASKCLUSTER_TMP_DIR}/${model_name_mmap} --scorer ${TASKCLUSTER_TMP_DIR}/kenlm.scorer --audio ${TASKCLUSTER_TMP_DIR}/${ldc93s1_sample_filename} --stream --extended 2>${TASKCLUSTER_TMP_DIR}/stderr | tail -n 1)
  status=$?
  set -e
  assert_correct_ldc93s1_prodtflitemodel "${phrase_pbmodel_withlm}" "$status" "${_bitrate}"
}
