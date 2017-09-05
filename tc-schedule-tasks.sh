#!/bin/bash

set -xe

PATH=/tmp:${PATH}
export PATH

wget https://github.com/taskcluster/slugid-go/releases/download/v1.0.0/slug-linux-amd64 -O /tmp/slug && chmod +x /tmp/slug

echo "python -c 'import sys; import json; import yaml; json.dump(yaml.load(sys.stdin), sys.stdout);'" > /tmp/yaml2json && chmod +x /tmp/yaml2json

TASK_PROVISIONER_ID="aws-provisioner-v1"
TASK_SCHEDULER_ID="taskcluster-github"
TASK_WORKER_TYPE="deepspeech-worker"

TASK_CREATED="$(date -u +'%Y-%m-%dT%H:%M:%S.000Z')"
TASK_DEADLINE="$(date -u +'%Y-%m-%dT%H:%M:%S.000Z' -d '1 day')"
TASK_EXPIRES="$(date -u +'%Y-%m-%dT%H:%M:%S.000Z' -d '7 days')"
ARTIFACTS_EXPIRES="$(date -u +'%Y-%m-%dT%H:%M:%S.000Z' -d '7 days')"

TASK_GROUP_ID=$(curl --silent http://taskcluster/queue/v1/task/${TASK_ID} | python -c 'import sys; import json; sys.stdout.write(json.load(sys.stdin)["taskGroupId"]);')

SYSTEM_ADD_USER='adduser --system --home /home/build-user build-user \&\& cd /home/build-user'
SYSTEM_DO_CLONE='sudo -H -u build-user /bin/bash /tmp/clone.sh'

TASK_ENV_VARS='DEEPSPEECH_ARTIFACTS_ROOT=${DEEPSPEECH_ARTIFACTS_ROOT} DEEPSPEECH_MODEL=${DEEPSPEECH_MODEL} TRAINING_TASK_ID=${TRAINING_TASK_ID}'

# Scheduling a training task
TRAINING_TASK_ID="$(slug)"
sed -e "s|{{ TASK_ID }}|${TASK_ID}|g" \
    -e "s|{{ TASK_GROUP_ID }}|${TASK_GROUP_ID}|g" \
    -e "s|{{ TASK_CREATED }}|${TASK_CREATED}|g" \
    -e "s|{{ TASK_DEADLINE }}|${TASK_DEADLINE}|g" \
    -e "s|{{ TASK_EXPIRES }}|${TASK_EXPIRES}|g" \
    -e "s|{{ TASK_ENV_VARS }}|${TASK_ENV_VARS}|g" \
    -e "s|{{ TASK_PROVISIONER_ID }}|${TASK_PROVISIONER_ID}|g" \
    -e "s|{{ TASK_SCHEDULER_ID }}|${TASK_SCHEDULER_ID}|g" \
    -e "s|{{ TASK_WORKER_TYPE }}|${TASK_WORKER_TYPE}|g" \
    -e "s|{{ ARTIFACTS_EXPIRES }}|${ARTIFACTS_EXPIRES}|g" \
    -e "s|{{ EVENT_HEAD_USER_EMAIL }}|${EVENT_HEAD_USER_EMAIL}|g" \
    -e "s|{{ EVENT_HEAD_REPO_URL }}|${EVENT_HEAD_REPO_URL}|g" \
    -e "s|{{ GITHUB_HEAD_REPO_URL }}|${GITHUB_HEAD_REPO_URL}|g" \
    -e "s|{{ GITHUB_HEAD_SHA }}|${GITHUB_HEAD_SHA}|g" \
    -e "s|{{ SYSTEM_ADD_USER }}|${SYSTEM_ADD_USER}|g" \
    -e "s|{{ SYSTEM_DO_CLONE }}|${SYSTEM_DO_CLONE}|g" \
    /home/build-user/DeepSpeech/ds/.tc.training.yml | yaml2json | curl -v -H 'Content-Type: application/json' -X PUT --data-binary @- http://taskcluster/queue/v1/task/${TRAINING_TASK_ID}

# Scheduling C++ deepspeech tests
sed -e "s|{{ TASK_ID }}|${TASK_ID}|g" \
    -e "s|{{ TRAINING_TASK_ID }}|${TRAINING_TASK_ID}|g" \
    -e "s|{{ TASK_GROUP_ID }}|${TASK_GROUP_ID}|g" \
    -e "s|{{ TASK_CREATED }}|${TASK_CREATED}|g" \
    -e "s|{{ TASK_DEADLINE }}|${TASK_DEADLINE}|g" \
    -e "s|{{ TASK_EXPIRES }}|${TASK_EXPIRES}|g" \
    -e "s|{{ TASK_ENV_VARS }}|${TASK_ENV_VARS}|g" \
    -e "s|{{ TASK_PROVISIONER_ID }}|${TASK_PROVISIONER_ID}|g" \
    -e "s|{{ TASK_SCHEDULER_ID }}|${TASK_SCHEDULER_ID}|g" \
    -e "s|{{ TASK_WORKER_TYPE }}|${TASK_WORKER_TYPE}|g" \
    -e "s|{{ ARTIFACTS_EXPIRES }}|${ARTIFACTS_EXPIRES}|g" \
    -e "s|{{ EVENT_HEAD_USER_EMAIL }}|${EVENT_HEAD_USER_EMAIL}|g" \
    -e "s|{{ EVENT_HEAD_REPO_URL }}|${EVENT_HEAD_REPO_URL}|g" \
    -e "s|{{ GITHUB_HEAD_REPO_URL }}|${GITHUB_HEAD_REPO_URL}|g" \
    -e "s|{{ GITHUB_HEAD_SHA }}|${GITHUB_HEAD_SHA}|g" \
    -e "s|{{ SYSTEM_ADD_USER }}|${SYSTEM_ADD_USER}|g" \
    -e "s|{{ SYSTEM_DO_CLONE }}|${SYSTEM_DO_CLONE}|g" \
    /home/build-user/DeepSpeech/ds/.tc.cpp-ds-tests.yml | yaml2json | curl -v -H 'Content-Type: application/json' -X PUT --data-binary @- http://taskcluster/queue/v1/task/$(slug)

# Scheduling Benchmark run
sed -e "s|{{ TASK_ID }}|${TASK_ID}|g" \
    -e "s|{{ TRAINING_TASK_ID }}|${TRAINING_TASK_ID}|g" \
    -e "s|{{ TASK_GROUP_ID }}|${TASK_GROUP_ID}|g" \
    -e "s|{{ TASK_CREATED }}|${TASK_CREATED}|g" \
    -e "s|{{ TASK_DEADLINE }}|${TASK_DEADLINE}|g" \
    -e "s|{{ TASK_EXPIRES }}|${TASK_EXPIRES}|g" \
    -e "s|{{ TASK_ENV_VARS }}|${TASK_ENV_VARS}|g" \
    -e "s|{{ TASK_PROVISIONER_ID }}|${TASK_PROVISIONER_ID}|g" \
    -e "s|{{ TASK_SCHEDULER_ID }}|${TASK_SCHEDULER_ID}|g" \
    -e "s|{{ TASK_WORKER_TYPE }}|${TASK_WORKER_TYPE}|g" \
    -e "s|{{ ARTIFACTS_EXPIRES }}|${ARTIFACTS_EXPIRES}|g" \
    -e "s|{{ EVENT_HEAD_USER_EMAIL }}|${EVENT_HEAD_USER_EMAIL}|g" \
    -e "s|{{ EVENT_HEAD_REPO_URL }}|${EVENT_HEAD_REPO_URL}|g" \
    -e "s|{{ GITHUB_HEAD_REPO_URL }}|${GITHUB_HEAD_REPO_URL}|g" \
    -e "s|{{ GITHUB_HEAD_SHA }}|${GITHUB_HEAD_SHA}|g" \
    -e "s|{{ SYSTEM_ADD_USER }}|${SYSTEM_ADD_USER}|g" \
    -e "s|{{ SYSTEM_DO_CLONE }}|${SYSTEM_DO_CLONE}|g" \
    /home/build-user/DeepSpeech/ds/.tc.benchmark-tests.yml | yaml2json | curl -v -H 'Content-Type: application/json' -X PUT --data-binary @- http://taskcluster/queue/v1/task/$(slug)

# Scheduling Python tests
for test_pyver in 2.7.13 3.4.6 3.5.3 3.6.2;
do
    sed -e "s|{{ TASK_ID }}|${TASK_ID}|g" \
        -e "s|{{ TRAINING_TASK_ID }}|${TRAINING_TASK_ID}|g" \
        -e "s|{{ TASK_GROUP_ID }}|${TASK_GROUP_ID}|g" \
        -e "s|{{ TASK_CREATED }}|${TASK_CREATED}|g" \
        -e "s|{{ TASK_DEADLINE }}|${TASK_DEADLINE}|g" \
        -e "s|{{ TASK_EXPIRES }}|${TASK_EXPIRES}|g" \
        -e "s|{{ TASK_ENV_VARS }}|${TASK_ENV_VARS}|g" \
        -e "s|{{ TASK_PROVISIONER_ID }}|${TASK_PROVISIONER_ID}|g" \
        -e "s|{{ TASK_SCHEDULER_ID }}|${TASK_SCHEDULER_ID}|g" \
        -e "s|{{ TASK_WORKER_TYPE }}|${TASK_WORKER_TYPE}|g" \
        -e "s|{{ ARTIFACTS_EXPIRES }}|${ARTIFACTS_EXPIRES}|g" \
        -e "s|{{ EVENT_HEAD_USER_EMAIL }}|${EVENT_HEAD_USER_EMAIL}|g" \
        -e "s|{{ EVENT_HEAD_REPO_URL }}|${EVENT_HEAD_REPO_URL}|g" \
        -e "s|{{ GITHUB_HEAD_REPO_URL }}|${GITHUB_HEAD_REPO_URL}|g" \
        -e "s|{{ GITHUB_HEAD_SHA }}|${GITHUB_HEAD_SHA}|g" \
        -e "s|{{ PYVER }}|${test_pyver}|g" \
        -e "s|{{ SYSTEM_ADD_USER }}|${SYSTEM_ADD_USER}|g" \
        -e "s|{{ SYSTEM_DO_CLONE }}|${SYSTEM_DO_CLONE}|g" \
        /home/build-user/DeepSpeech/ds/.tc.python-tests.yml | yaml2json | curl -v -H 'Content-Type: application/json' -X PUT --data-binary @- http://taskcluster/queue/v1/task/$(slug)
done;

# Scheduling NodeJS tests
## Disable 7.x and 8.x for now
for test_nodever in 4.x 5.x 6.x;
do
    sed -e "s|{{ TASK_ID }}|${TASK_ID}|g" \
        -e "s|{{ TRAINING_TASK_ID }}|${TRAINING_TASK_ID}|g" \
        -e "s|{{ TASK_GROUP_ID }}|${TASK_GROUP_ID}|g" \
        -e "s|{{ TASK_CREATED }}|${TASK_CREATED}|g" \
        -e "s|{{ TASK_DEADLINE }}|${TASK_DEADLINE}|g" \
        -e "s|{{ TASK_EXPIRES }}|${TASK_EXPIRES}|g" \
        -e "s|{{ TASK_ENV_VARS }}|${TASK_ENV_VARS}|g" \
        -e "s|{{ TASK_PROVISIONER_ID }}|${TASK_PROVISIONER_ID}|g" \
        -e "s|{{ TASK_SCHEDULER_ID }}|${TASK_SCHEDULER_ID}|g" \
        -e "s|{{ TASK_WORKER_TYPE }}|${TASK_WORKER_TYPE}|g" \
        -e "s|{{ ARTIFACTS_EXPIRES }}|${ARTIFACTS_EXPIRES}|g" \
        -e "s|{{ EVENT_HEAD_USER_EMAIL }}|${EVENT_HEAD_USER_EMAIL}|g" \
        -e "s|{{ EVENT_HEAD_REPO_URL }}|${EVENT_HEAD_REPO_URL}|g" \
        -e "s|{{ GITHUB_HEAD_REPO_URL }}|${GITHUB_HEAD_REPO_URL}|g" \
        -e "s|{{ GITHUB_HEAD_SHA }}|${GITHUB_HEAD_SHA}|g" \
        -e "s|{{ NODEVER }}|${test_nodever}|g" \
        -e "s|{{ SYSTEM_ADD_USER }}|${SYSTEM_ADD_USER}|g" \
        -e "s|{{ SYSTEM_DO_CLONE }}|${SYSTEM_DO_CLONE}|g" \
        /home/build-user/DeepSpeech/ds/.tc.node-tests.yml | yaml2json | curl -v -H 'Content-Type: application/json' -X PUT --data-binary @- http://taskcluster/queue/v1/task/$(slug)
done;
