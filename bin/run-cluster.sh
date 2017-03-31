#!/bin/bash

if [ ! -f DeepSpeech.py ]; then
    echo "Please make sure you run this from DeepSpeech's top level directory."
    exit 1
fi;

ps_count=1
worker_count=2
gpu_count=0
script="python -u DeepSpeech.py"

if [[ $1 == "--help" ]]; then
  echo "Usage: run-cluster.sh [--help] [--script script] [p:w:g] <arg>*"
  echo ""
  echo "--help      print this help message"
  echo "--script    run the provided script instead of DeepSpeech.py"
  echo "p           number of local parameter servers"
  echo "w           number of local workers"
  echo "g           number of local GPUs per worker"
  echo "<arg>*      remaining parameters will be forwarded to DeepSpeech.py or a provided script"
  echo
  echo "Example usage - The following example will create a local DeepSpeech.py cluster"
  echo "with 1 parameter server, and 2 workers with 1 GPU each:"
  echo "$ run-cluster.sh 1:2:1 --epoch 10"
  echo
  exit 0
fi

if [[ $1 == "--script" ]]; then
  shift 1
  script=$1
  shift 1
  echo "Using script $script..."
fi

if [[ $1 =~ ([0-9]+):([0-9]+):([0-9]+) ]]; then
  ps_count=${BASH_REMATCH[1]}
  worker_count=${BASH_REMATCH[2]}
  gpu_count=${BASH_REMATCH[3]}
  shift 1
fi

echo "Starting cluster with $ps_count parameter servers and $worker_count workers with $gpu_count GPUs each..."

# Generating the parameter server addresses
index=0
while [ "$index" -lt "$ps_count" ]
do
  ps_hosts[$index]="localhost:$((index + 2000))"
  ((index++))
done
ps_hosts=$(printf ",%s" "${ps_hosts[@]}")
ps_hosts=${ps_hosts:1}

# Generating the worker addresses
index=0
while [ "$index" -lt "$worker_count" ]
do
  worker_hosts[$index]="localhost:$((index + 3000))"
  ((index++))
done
worker_hosts=$(printf ",%s" "${worker_hosts[@]}")
worker_hosts=${worker_hosts:1}


# Starting the parameter servers
index=0
while [ "$index" -lt "$ps_count" ]
do
  CUDA_VISIBLE_DEVICES="" $script --ps_hosts $ps_hosts --worker_hosts $worker_hosts --job_name=ps --task_index=$index "$@" 2>&1 | sed 's/^/[ps     '"$index"'] /' &
  echo "Started ps $index"
  ((index++))
done

# Starting the workers
start=0
index=0
while [ "$index" -lt "$worker_count" ]
do
    stop=$((start+gpu_count-1))
    # Creating a comma delimited number sequence from $start to $end
    cvd=`seq -s, $start $stop`
    CUDA_VISIBLE_DEVICES=$cvd $script --ps_hosts $ps_hosts --worker_hosts $worker_hosts --job_name=worker --task_index=$index "$@" 2>&1 | sed 's/^/[worker '"$index"'] /' &
    start=$((start+gpu_count))
  echo "Started worker $index"
  ((index++))
done

# If we are forced to quit, we kill all ramining jobs/servers
function quit {
  echo
  echo "Killing whole process group - the hard way..."
  kill -KILL -$$
}
trap quit SIGINT SIGTERM

# Waiting for all running jobs to join
while [ `jobs -rp | wc -l` -gt 0 ]; do sleep 1; done
