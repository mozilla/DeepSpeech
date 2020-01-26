#!/bin/sh

set -xe

ru_dir="./data/ru"
ru_csv="${ru_dir}/ru.csv"

epoch_count=$1

if [ ! -f "${ru_dir}/ru.csv" ]; then
    echo "Downloading and preprocessing Russian example data, saving in ${ru_dir}."
    python -u bin/import_ru.py ${ru_dir}
fi;

# Force only one visible device because we have a single-sample dataset
# and when trying to run on multiple devices (like GPUs), this will break
export CUDA_VISIBLE_DEVICES=0

# WORKS: #

echo "# CUDNN FINE"
python -u DeepSpeech.py --noshow_progressbar --noearly_stop \
       --fine_tune \
       --alphabet_config_path "${ru_dir}/alphabet.ru" \
       --load "transfer" --drop_source_layers 1 \
       --train_files  "${ru_dir}/ru.csv" --train_batch_size 1  \
       --dev_files  "${ru_dir}/ru.csv" --dev_batch_size 1 \
       --test_files  "${ru_dir}/ru.csv" --test_batch_size 1 \
       --checkpoint_dir '/tmp/ckpt/transfer' --epochs 5 \
       --cudnn_source_model_checkpoint_dir "/home/josh/Downloads/deepspeech-0.6.1-checkpoint/" | tee /tmp/transfer.log

echo "# CUDNN FROZEN"
python -u DeepSpeech.py --noshow_progressbar --noearly_stop \
       --alphabet_config_path "${ru_dir}/alphabet.ru" \
       --load "transfer" --drop_source_layers 1 \
       --train_files  "${ru_dir}/ru.csv" --train_batch_size 1  \
       --dev_files  "${ru_dir}/ru.csv" --dev_batch_size 1 \
       --test_files  "${ru_dir}/ru.csv" --test_batch_size 1 \
       --checkpoint_dir '/tmp/ckpt/transfer' --epochs 5 \
       --cudnn_source_model_checkpoint_dir "/home/josh/Downloads/deepspeech-0.6.1-checkpoint/" | tee /tmp/transfer.log

echo "# REG RNN FINE"
python -u DeepSpeech.py --noshow_progressbar --noearly_stop \
       --fine_tune \
       --alphabet_config_path "${ru_dir}/alphabet.ru" \
       --load "transfer" --drop_source_layers 3 \
       --train_files  "${ru_dir}/ru.csv" --train_batch_size 1  \
       --dev_files  "${ru_dir}/ru.csv" --dev_batch_size 1 \
       --test_files  "${ru_dir}/ru.csv" --test_batch_size 1 \
       --checkpoint_dir '/tmp/ckpt/transfer' --epochs 5 \
       --source_model_checkpoint_dir "/home/josh/Downloads/deepspeech-0.6.1-checkpoint/" | tee /tmp/transfer.log


echo "# REG RNN FROZEN"
python -u DeepSpeech.py --noshow_progressbar --noearly_stop \
       --alphabet_config_path "${ru_dir}/alphabet.ru" \
       --load "transfer" --drop_source_layers 3 \
       --train_files  "${ru_dir}/ru.csv" --train_batch_size 1  \
       --dev_files  "${ru_dir}/ru.csv" --dev_batch_size 1 \
       --test_files  "${ru_dir}/ru.csv" --test_batch_size 1 \
       --checkpoint_dir '/tmp/ckpt/transfer' --epochs 5 \
       --source_model_checkpoint_dir "/home/josh/Downloads/deepspeech-0.6.1-checkpoint/" | tee /tmp/transfer.log
