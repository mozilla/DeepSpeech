#!/bin/sh

set -xe

ldc93s1_dir="./data/ldc93s1-tc"
ldc93s1_csv="${ldc93s1_dir}/ldc93s1.csv"

epoch_count=$1

if [ ! -f "${ldc93s1_dir}/ldc93s1.csv" ]; then
    echo "Downloading and preprocessing LDC93S1 example data, saving in ${ldc93s1_dir}."
    python -u bin/import_ldc93s1.py ${ldc93s1_dir}
fi;

python -u DeepSpeech.py --noshow_progressbar \
  --train_files ${ldc93s1_csv} --train_batch_size 1 \
  --dev_files ${ldc93s1_csv} --dev_batch_size 1 \
  --test_files ${ldc93s1_csv} --test_batch_size 1 \
  --n_hidden 494 --epoch -1 --random_seed 4567 --default_stddev 0.046875 \
  --max_to_keep 1 --checkpoint_dir '/tmp/ckpt' \
  --learning_rate 0.001 --dropout_rate 0.05 \
  --lm_binary_path 'data/smoke_test/vocab.pruned.lm' \
  --lm_trie_path 'data/smoke_test/vocab.trie' | tee /tmp/resume.log

if ! grep "Training of Epoch $epoch_count" /tmp/resume.log; then
  echo "Did not resume training from checkpoint"
  exit 1
else
  exit 0
fi
