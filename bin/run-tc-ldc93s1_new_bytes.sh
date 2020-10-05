#!/bin/sh

set -xe

ldc93s1_dir="./data/smoke_test"
ldc93s1_csv="${ldc93s1_dir}/ldc93s1.csv"

epoch_count=$1
audio_sample_rate=$2

if [ ! -f "${ldc93s1_dir}/ldc93s1.csv" ]; then
    echo "Downloading and preprocessing LDC93S1 example data, saving in ${ldc93s1_dir}."
    python -u bin/import_ldc93s1.py ${ldc93s1_dir}
fi;

# Force only one visible device because we have a single-sample dataset
# and when trying to run on multiple devices (like GPUs), this will break
export CUDA_VISIBLE_DEVICES=0

python -u DeepSpeech.py --noshow_progressbar --noearly_stop \
  --train_files ${ldc93s1_csv} --train_batch_size 1 \
  --feature_cache '/tmp/ldc93s1_cache' \
  --dev_files ${ldc93s1_csv} --dev_batch_size 1 \
  --test_files ${ldc93s1_csv} --test_batch_size 1 \
  --n_hidden 100 --epochs $epoch_count \
  --max_to_keep 1 --checkpoint_dir '/tmp/ckpt_bytes' \
  --learning_rate 0.001 --dropout_rate 0.05  --export_dir '/tmp/train_bytes' \
  --scorer_path 'data/smoke_test/pruned_lm.bytes.scorer' \
  --audio_sample_rate ${audio_sample_rate} \
  --bytes_output_mode
