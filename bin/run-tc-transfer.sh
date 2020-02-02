#!/bin/sh
'''
This bash script is for running minimum working examples
of transfer learning for continuous integration tests
to be run on taskcluster.
'''
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



echo "##### Train ENGLISH model and transfer to RUSSIAN #####"
echo "##### while iterating over loading logic #####"

for LOAD in 'init' 'last' 'auto'; do
    python -u DeepSpeech.py --noshow_progressbar --noearly_stop\
           --alphabet_config_path "./data/alphabet.txt" \
           --load "$LOAD" \
           --train_files  "./data/ldc93s1/ldc93s1.csv" --train_batch_size 1  \
           --dev_files  "./data/ldc93s1/ldc93s1.csv" --dev_batch_size 1 \
           --test_files  "./data/ldc93s1/ldc93s1.csv" --test_batch_size 1 \
           --save_checkpoint_dir '/tmp/ckpt/transfer/eng-cudnn' \
           --load_checkpoint_dir '/tmp/ckpt/transfer/eng-cudnn' \
           --n_hidden 100 \
           --epochs 10 \
           "$@"

    python -u DeepSpeech.py --noshow_progressbar --noearly_stop\
           --drop_source_layers 1 \
           --alphabet_config_path "${ru_dir}/alphabet.ru" \
           --load 'last' \
           --train_files  "${ru_dir}/ru.csv" --train_batch_size 1  \
           --dev_files  "${ru_dir}/ru.csv" --dev_batch_size 1 \
           --test_files  "${ru_dir}/ru.csv" --test_batch_size 1 \
           --save_checkpoint_dir '/tmp/ckpt/transfer/ru-cudnn' \
           --load_checkpoint_dir '/tmp/ckpt/transfer/eng-cudnn' \
           --n_hidden 100 \
           --epochs 10 \
           "$@"
done


echo "##### Train ENGLISH model and transfer to RUSSIAN #####"
echo "##### while iterating over loading logic with CUDNN #####"

for LOAD in 'init' 'last' 'auto'; do
    python -u DeepSpeech.py --noshow_progressbar --noearly_stop\
           --train_cudnn\
           --alphabet_config_path "./data/alphabet.txt" \
           --load "$LOAD" \
           --train_files  "./data/ldc93s1/ldc93s1.csv" --train_batch_size 1  \
           --dev_files  "./data/ldc93s1/ldc93s1.csv" --dev_batch_size 1 \
           --test_files  "./data/ldc93s1/ldc93s1.csv" --test_batch_size 1 \
           --save_checkpoint_dir '/tmp/ckpt/transfer/eng-cudnn' \
           --load_checkpoint_dir '/tmp/ckpt/transfer/eng-cudnn' \
           --n_hidden 100 \
           --epochs 10 \
           "$@"

    python -u DeepSpeech.py --noshow_progressbar --noearly_stop\
           --load_cudnn\
           --drop_source_layers 1 \
           --alphabet_config_path "${ru_dir}/alphabet.ru" \
           --load 'last' \
           --train_files  "${ru_dir}/ru.csv" --train_batch_size 1  \
           --dev_files  "${ru_dir}/ru.csv" --dev_batch_size 1 \
           --test_files  "${ru_dir}/ru.csv" --test_batch_size 1 \
           --save_checkpoint_dir '/tmp/ckpt/transfer/ru-cudnn' \
           --load_checkpoint_dir '/tmp/ckpt/transfer/eng-cudnn' \
           --n_hidden 100 \
           --epochs 10 \
           "$@"
done
