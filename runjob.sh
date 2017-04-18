BUCKET_NAME="gs://deep_speech"
TRAIN_DATA="$BUCKET_NAME/DeepSpeech/data_files/data/ldc93s1/ldc93s1.csv"
DEV_DATA="$BUCKET_NAME/DeepSpeech/data_files/data/ldc93s1/ldc93s1.csv"
TEST_DATA="$BUCKET_NAME/DeepSpeech/data_files/data/ldc93s1/ldc93s1.csv"
checkpoint_dir="$BUCKET_NAME/DeepSpeech/checkpoint"
now=$(date +"%Y%m%d_%H%M%S")
JOB_NAME="job_$now"
PACKAGE_PATH="trainer"
JOB_DIRECTORY="gs://deep_speech/output"
DEPENDENCY_PATH="~/Varunproject/DeepSpeech/util/"
MODULE_NAME="trainer.task"
STAGING_BUCKET="gs://deep_speech/staging_folder"
JOB_DIR="gs://deep_speech/output"
REGION="us-central1"
PACKAGES="trainer-0.1-py2-none-any.whl"
RUNTIME_VERSION="1.0"

gcloud ml-engine jobs submit training $JOB_NAME \
--job-dir $JOB_DIRECTORY \
--package-path $PACKAGE_PATH \
--module-name $MODULE_NAME \
--region $REGION \
--packages $PACKAGES \
-- \
--train_files $TRAIN_DATA \
--dev_files $DEV_DATA \
--test_files $TEST_DATA \
--train_batch_size 1  \
--dev_batch_size 1 \
--test_batch_size 1 \
--n_hidden 494 \
--epoch 50 \
--checkpoint_dir "$checkpoint_dir" \
