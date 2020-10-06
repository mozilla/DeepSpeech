#!/bin/sh

set -xe

ldc93s1_dir="./data/smoke_test"
ldc93s1_csv="${ldc93s1_dir}/ldc93s1.csv"

audio_sample_rate=$1

if [ ! -f "${ldc93s1_dir}/ldc93s1.csv" ]; then
    echo "Downloading and preprocessing LDC93S1 example data, saving in ${ldc93s1_dir}."
    python -u bin/import_ldc93s1.py ${ldc93s1_dir}
fi;

# Force only one visible device because we have a single-sample dataset
# and when trying to run on multiple devices (like GPUs), this will break
export CUDA_VISIBLE_DEVICES=0

python -u DeepSpeech.py --noshow_progressbar \
  --n_hidden 100 \
  --checkpoint_dir '/tmp/ckpt_bytes' \
  --export_dir '/tmp/train_bytes_tflite' \
  --scorer_path 'data/smoke_test/pruned_lm.bytes.scorer' \
  --bytes_output_mode \
  --audio_sample_rate ${audio_sample_rate} \
  --export_tflite
