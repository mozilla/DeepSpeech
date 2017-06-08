#LDC93S1.txt  LDC93S1.wav  __init__.py  ldc93s1.csv
BUCKET_NAME="~/Varunproject"
TRAIN_DATA="~/Varunproject/DeepSpeech/data/ldc93s1/ldc93s1.csv"
DEV_DATA="~/Varunproject/DeepSpeech/data/ldc93s1/ldc93s1.csv"
TEST_DATA="~/Varunproject/DeepSpeech/data/ldc93s1/ldc93s1.csv"
checkpoint_dir="~/Varunproject/DeepSpeech/checkpoint"
gcloud ml-engine local train \
    --module-name trainer.task \
    --package-path trainer/ \
    --\
    --train_files $TRAIN_DATA \
--dev_files $DEV_DATA \
--test_files $TEST_DATA \
--train_batch_size 1  \
--dev_batch_size 1 \
--test_batch_size 1 \
--n_hidden 494 \
--epoch 50 \
--checkpoint_dir "$checkpoint_dir" \
