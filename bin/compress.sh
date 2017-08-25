#!/bin/sh
DATASET=$1
CODEBOOK_DIR="codebook"
NUM=5
if [ ! $DATASET ]; then
    echo "Usage ./bin/prune.sh  [ldc93s1 | ted] "
    exit
fi;

EXTRA_PARAM=$@
# Removing the data set from the argument list
EXTRA_PARAM="$(echo $EXTRA_PARAM | cut -d ' ' --complement -s -f1)"
if [ ! -d "${COMPUTE_DATA_DIR}" ]; then
    COMPUTE_DATA_DIR="data"
fi;

if [ "$DATASET" = "ldc93s1" ]; then
    DATA_DIR='./data/ldc93s1'
    TRAIN_FILE='data/ldc93s1/ldc93s1.csv'
    DEV_FILE='data/ldc93s1/ldc93s1.csv'
    TEST_FILE='data/ldc93s1/ldc93s1.csv'
    IMPORTER='bin/import_ldc93s1.py'
    EPOCH=70
    N_HIDDEN=494
    TRAIN_BATCH_SIZE=1
    DEV_BATCH_SIZE=1
    PR_THRESH_WEIGHT=0.0001
    TEST_BATCH_SIZE=1

elif [ "$DATASET" = "ted" ]; then
    DATA_DIR='./data/ted'
    TRAIN_FILE="$COMPUTE_DATA_DIR/ted-train.csv"
    DEV_FILE="$COMPUTE_DATA_DIR/ted-dev.csv"
    TEST_FILE="$COMPUTE_DATA_DIR/ted-test.csv"
    IMPORTER='bin/import_ted.py'
    EPOCH=10
    N_HIDDEN=2048
    TRAIN_BATCH_SIZE=16
    DEV_BATCH_SIZE=8
    TEST_BATCH_SIZE=8
    PR_THRESH_WEIGHT=0.00001
    EXTRA_PARAM="--dropout_rate 0.30 --default_stddev 0.046875 --learning_rate 0.0001 --validtion_step 1 --display_step 10 $EXTRA_PARAM"

fi;

set -xe
if [ ! -f DeepSpeech.py ]; then
    echo "Please make sure you run this from DeepSpeech's top level directory."
    exit 1
fi;

if [ ! -f "$TRAIN_FILE" ]; then
    #echo "Downloading and preprocessing LDC93S1 example data, saving in ./data/ldc93s1."
    python -u $IMPORTER $DATA_DIR
fi;

if [ -d "${COMPUTE_KEEP_DIR}" ]; then
    checkpoint_dir=$COMPUTE_KEEP_DIR
else
    checkpoint_dir=$(python -c "from xdg import BaseDirectory as xdg; print(xdg.save_data_path(\"deepspeech/$DATASET\"))")
fi

codebook_dir=$(python -c "from xdg import BaseDirectory as xdg; print(xdg.save_data_path(\"deepspeech/$DATASET/$CODEBOOK_DIR\"))")

train()
{
    python -u DeepSpeech.py \
        --train_files $TRAIN_FILE \
        --dev_files $DEV_FILE \
        --test_files $TEST_FILE \
        --train_batch_size $TRAIN_BATCH_SIZE \
        --dev_batch_size $DEV_BATCH_SIZE \
        --test_batch_size $TEST_BATCH_SIZE \
        --n_hidden $N_HIDDEN \
        --epoch $EPOCH \
        --checkpoint_dir "$checkpoint_dir" \
        --weight_sharing True \
        --pruning True \
        --codebook_dir "$codebook_dir" \
        --prune_threshold_weight $PR_THRESH_WEIGHT \
        $EXTRA_PARAM
}


infer()
{
    python -u inference.py \
        --dataset $DATASET \
        --codebook_dir "$codebook_dir" \
        --n_hidden $N_HIDDEN \
        "$EXTRA_PARAM"
}

main()
{
    echo "Initiate the training sequence with deep compressio enabled"
    train
    echo "Perform the inference on trained model with deep compression enabled"
    infer
}

main
