#!/bin/sh
# This bash script is for running minimum working examples
# of transfer learning for continuous integration tests
# to be run on Taskcluster.
set -xe

ru_dir="./data/smoke_test/russian_sample_data"
ru_csv="${ru_dir}/ru.csv"

ldc93s1_dir="./data/smoke_test"
ldc93s1_csv="${ldc93s1_dir}/ldc93s1.csv"

if [ ! -f "${ldc93s1_dir}/ldc93s1.csv" ]; then
    echo "Downloading and preprocessing LDC93S1 example data, saving in ${ldc93s1_dir}."
    python -u bin/import_ldc93s1.py ${ldc93s1_dir}
fi;

# Force only one visible device because we have a single-sample dataset
# and when trying to run on multiple devices (like GPUs), this will break
export CUDA_VISIBLE_DEVICES=0

# Force UTF-8 output
export PYTHONIOENCODING=utf-8

echo "##### Train ENGLISH model and transfer to RUSSIAN #####"
echo "##### while iterating over loading logic #####"

for LOAD in 'init' 'last' 'auto'; do
    echo "########################################################"
    echo "#### Train ENGLISH model with just --checkpoint_dir ####"
    echo "########################################################"
    python -u DeepSpeech.py --noshow_progressbar --noearly_stop \
       --alphabet_config_path "./data/alphabet.txt" \
       --load_train "$LOAD" \
       --train_files  "${ldc93s1_csv}" --train_batch_size 1  \
       --dev_files  "${ldc93s1_csv}" --dev_batch_size 1 \
       --test_files  "${ldc93s1_csv}" --test_batch_size 1 \
       --scorer_path '' \
       --checkpoint_dir '/tmp/ckpt/transfer/eng' \
       --n_hidden 100 \
       --epochs 10

    echo "##############################################################################"
    echo "#### Train ENGLISH model with --save_checkpoint_dir --load_checkpoint_dir ####"
    echo "##############################################################################"
    python -u DeepSpeech.py --noshow_progressbar --noearly_stop \
           --alphabet_config_path "./data/alphabet.txt" \
           --load_train "$LOAD" \
           --train_files  "${ldc93s1_csv}" --train_batch_size 1  \
           --dev_files  "${ldc93s1_csv}" --dev_batch_size 1 \
           --test_files  "${ldc93s1_csv}" --test_batch_size 1 \
           --save_checkpoint_dir '/tmp/ckpt/transfer/eng' \
           --load_checkpoint_dir '/tmp/ckpt/transfer/eng' \
           --scorer_path '' \
           --n_hidden 100 \
           --epochs 10

    echo "####################################################################################"
    echo "#### Transfer to RUSSIAN model with --save_checkpoint_dir --load_checkpoint_dir ####"
    echo "####################################################################################"
    python -u DeepSpeech.py --noshow_progressbar --noearly_stop \
           --drop_source_layers 1 \
           --alphabet_config_path "${ru_dir}/alphabet.ru" \
           --load_train 'last' \
           --train_files  "${ru_csv}" --train_batch_size 1  \
           --dev_files  "${ru_csv}" --dev_batch_size 1 \
           --save_checkpoint_dir '/tmp/ckpt/transfer/ru' \
           --load_checkpoint_dir '/tmp/ckpt/transfer/eng' \
           --scorer_path '' \
           --n_hidden 100 \
           --epochs 10

    # Test transfer learning checkpoint
    python -u evaluate.py --noshow_progressbar \
           --test_files  "${ru_csv}" --test_batch_size 1 \
           --alphabet_config_path "${ru_dir}/alphabet.ru" \
           --load_checkpoint_dir '/tmp/ckpt/transfer/ru' \
           --scorer_path '' \
           --n_hidden 100
done
