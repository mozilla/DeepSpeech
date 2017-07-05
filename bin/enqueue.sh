#!/bin/bash

if [ ! -f DeepSpeech.py ]; then
  echo "Please make sure you run this from DeepSpeech's top level directory."
  exit 1
fi

GLOBAL_LOG="\\/dev\\/null" # by default there is no global logging
NODES=2 # to be adjusted
GPUS=8 # to be adjusted
HASH=`git rev-parse HEAD`
TO_RUN=run-`date +%Y-%m-%d-%H-%M` # default run name is based on user's name and current date time

runs_dir="/data/runs"

# parsing parameters
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    -h|--help)
    echo "Usage: enqueue.sh [--help] [-g|--gpus N] [-n|--nodes N] [-l|-log file] [-c|--continue run] [run]"
    echo ""
    echo "--help            print this help message"
    echo "--gpus N          allocates N gpus per node"
    echo "--nodes N         allocates N nodes"
    echo "--log file        if set, global logs will get appended to the provided file"
    echo "--continue run    continue a former run by copying its transitional content into the new run directory"
    echo "run               name of or path to a run directory - if just a name, it will be created at $runs_dir/<run>"
    echo "                  defaults to user's name plus current date/time"
    echo
    exit 0
    ;;
    -g|--gpus)
    GPUS="$2"
    shift
    ;;
    -n|--nodes)
    NODES="$2"
    shift
    ;;
    -l|--log)
    GLOBAL_LOG=$(sed 's/[\/&]/\\&/g' <<< $2) # path-slashes are to be escaped
    shift
    ;;
    -c|--continue)
    TO_CONTINUE="$2"
    shift
    ;;
    *)
    TO_RUN="$1"
    ;;
  esac
  shift # past argument or value
done



# creates a run directory path from a run name given by $1
# if run name is a directory, it's treated relative to current directory
# if just a name, it will be treated relative to $runs_dir
function to_run_dir {
  local run_dir
  if [[ $1 == .*\/.* ]]; then
    if [[ $1 == \/.* ]]; then
      run_dir=$1
    else
      run_dir=$runs_dir/$1
    fi
  else
    run_dir=$runs_dir/$1
  fi
  echo ${run_dir//\/\//\/}
}

# run directory from which "keep" data is to be copied over into the new run directory
if [ "$TO_CONTINUE" ]; then
  CONTINUE_DIR=$(to_run_dir "$TO_CONTINUE")
  if [ ! -d "$CONTINUE_DIR/results/keep" ]; then
    echo "Cannot continue former run. Directory $CONTINUE_DIR/results/keep doesn't exist."
    exit 1
  fi
fi

# preparing run directory
RUN_DIR=$(to_run_dir "$TO_RUN")
echo "Creating run directory $RUN_DIR..."
if [ -d "$RUN_DIR" ]; then
  echo "Run directory $RUN_DIR already exists."
  exit 1
elif [ -f "$RUN_DIR" ]; then
  echo "Run directory path $RUN_DIR is already a file."
  exit 1
fi
mkdir -p "$RUN_DIR/src"
mkdir -p "$RUN_DIR/results"

# copying local tree into "src" sub-directory of run directory
# excluding .git
rsync -av . "$RUN_DIR/src" --exclude=/.git

# copying over "keep" data from continue run directory
if [ "$CONTINUE_DIR" ]; then
    cp -rf "$CONTINUE_DIR/results/keep" "$RUN_DIR/results/"
fi

mkdir -p "$RUN_DIR/results/keep"

# patch-creating job.sbatch file from job-template with our parameters
sed \
    -e "s/__ID__/$HASH/g" \
    -e "s/__NAME__/$TO_RUN/g" \
    -e "s/__GLOBAL_LOG__/$GLOBAL_LOG/g" \
    -e "s/__NODES__/$NODES/g" \
    -e "s/__GPUS__/$GPUS/g" \
    bin/job-template.sbatch > $RUN_DIR/job.sbatch

# enqueuing the new job description and run directory
sbatch -D $RUN_DIR $RUN_DIR/job.sbatch
