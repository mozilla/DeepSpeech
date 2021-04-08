#!/bin/sh

set -xe

ldc93s1_dir="./data/smoke_test"
ldc93s1_csv="${ldc93s1_dir}/ldc93s1.csv"

if [ ! -f "${ldc93s1_dir}/ldc93s1.csv" ]; then
    echo "Downloading and preprocessing LDC93S1 example data, saving in ${ldc93s1_dir}."
    python -u bin/import_ldc93s1.py ${ldc93s1_dir}
fi;

# Force only one visible device because we have a single-sample dataset
# and when trying to run on multiple devices (like GPUs), this will break
export CUDA_VISIBLE_DEVICES=0

python -u DeepSpeech.py --noshow_progressbar --noearly_stop \
  --train_files ${ldc93s1_csv} --train_batch_size 1 \
  --scorer "" \
  --augment dropout \
  --augment pitch \
  --augment tempo \
  --augment warp \
  --augment time_mask \
  --augment frequency_mask \
  --augment add \
  --augment multiply \
  --n_hidden 100 \
  --epochs 1
