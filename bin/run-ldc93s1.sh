#!/bin/sh
set -xe
if [ ! -f DeepSpeech.py ]; then
    echo "Please make sure you run this from DeepSpeech's top level directory."
    exit 1
fi;

if [ ! -f "data/ldc93s1/ldc93s1.csv" ]; then
    echo "Downloading and preprocessing LDC93S1 example data, saving in ./data/ldc93s1."
    python -u bin/import_ldc93s1.py ./data/ldc93s1
fi;

if [ -d "${COMPUTE_KEEP_DIR}" ]; then
    checkpoint_dir=$COMPUTE_KEEP_DIR
else
    checkpoint_dir=$(python -c 'from xdg import BaseDirectory as xdg; print(xdg.save_data_path("deepspeech/ldc93s1"))')
fi

# Force only one visible device because we have a single-sample dataset
# and when trying to run on multiple devices (like GPUs), this will break
export CUDA_VISIBLE_DEVICES=0

# # echo "################## INIT HERE ##############################"
# # python -u DeepSpeech.py --noshow_progressbar \
# #        --drop_source_layers 2 \
# #        --load 'init' \
# #        --train_files data/ldc93s1/ldc93s1.csv \
# #        --test_files data/ldc93s1/ldc93s1.csv \
# #        --train_batch_size 1 \
# #        --test_batch_size 1 \
# #        --n_hidden 100 \
# #        --epochs 200 \
# #        --checkpoint_dir "$checkpoint_dir" \
# #        "$@"

# echo "################## LAST HERE ##############################"
# python -u DeepSpeech.py --noshow_progressbar \
#        --drop_source_layers 2 \
#        --load 'last' \
#        --train_files data/ldc93s1/ldc93s1.csv \
#        --test_files data/ldc93s1/ldc93s1.csv \
#        --train_batch_size 1 \
#        --test_batch_size 1 \
#        --n_hidden 100 \
#        --epochs 200 \
#        --checkpoint_dir "$checkpoint_dir" \
#        "$@"

# echo "################## AUTO HERE ##############################"
# python -u DeepSpeech.py --noshow_progressbar \
#        --drop_source_layers 2 \
#        --load 'auto' \
#        --train_files data/ldc93s1/ldc93s1.csv \
#        --test_files data/ldc93s1/ldc93s1.csv \
#        --train_batch_size 1 \
#        --test_batch_size 1 \
#        --n_hidden 100 \
#        --epochs 200 \
#        --checkpoint_dir "$checkpoint_dir" \
#        "$@"


# echo "################## INIT CUDNN ##############################"

# python -u DeepSpeech.py --noshow_progressbar \
#        --load 'init' \
#        --train_files data/ldc93s1/ldc93s1.csv \
#        --test_files data/ldc93s1/ldc93s1.csv \
#        --train_batch_size 1 \
#        --test_batch_size 1 \
#        --n_hidden 100 \
#        --epochs 200 \
#        --cudnn_checkpoint "$checkpoint_dir" \
#        "$@"


# echo "################## BEST CUDNN ##############################"

# python -u DeepSpeech.py --noshow_progressbar \
#        --load 'last' \
#        --train_files data/ldc93s1/ldc93s1.csv \
#        --test_files data/ldc93s1/ldc93s1.csv \
#        --train_batch_size 1 \
#        --test_batch_size 1 \
#        --n_hidden 100 \
#        --epochs 200 \
#        --cudnn_checkpoint "$checkpoint_dir" \
#        "$@"

echo "################## AUTO CUDNN ##############################"

python -u DeepSpeech.py --noshow_progressbar \
       --drop_source_layers 2 \
       --load 'auto' \
       --train_files data/ldc93s1/ldc93s1.csv \
       --test_files data/ldc93s1/ldc93s1.csv \
       --train_batch_size 1 \
       --test_batch_size 1 \
       --n_hidden 100 \
       --epochs 200 \
       --cudnn_checkpoint "$checkpoint_dir" \
       "$@"


echo "################## BEST CUDNN ##############################"

python -u DeepSpeech.py --noshow_progressbar \
       --drop_source_layers 2 \
       --load 'best' \
       --train_files data/ldc93s1/ldc93s1.csv \
       --test_files data/ldc93s1/ldc93s1.csv \
       --train_batch_size 1 \
       --test_batch_size 1 \
       --n_hidden 100 \
       --epochs 200 \
       --cudnn_checkpoint "$checkpoint_dir" \
       "$@"
