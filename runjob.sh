TRAIN_DATA="gs://deep_speech/data_files/data/ldc93s1/ldc93s1.csv"
DEV_DATA="gs://deep_speech/data_files/data/ldc93s1/ldc93s1.csv"
TEST_DATA="gs://deep_speech/data_files/data/ldc93s1/ldc93s1.csv"
checkpoint_dir="gs://deep_speech/checkpoint"
now=$(date +"%Y%m%d_%H%M%S")
JOB_NAME="job_$now"
PACKAGE_PATH="trainer/"


gcloud ml-engine jobs submit training $JOB_NAME \
--module-name trainer.task \
--package-path trainer/ \
--staging-bucket gs://deep_speech \
--job-dir gs://deep_speech/output \
--region us-central1 \
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
