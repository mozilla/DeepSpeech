#!/bin/bash

if [ ! -f DeepSpeech.py ]; then
    echo "Please make sure you run this from DeepSpeech's top level directory."
    exit 1
fi;

node_count=2
gpu_count=0
script="python -u DeepSpeech.py"

if [[ $1 == "--help" ]]; then
  echo "Usage: run-cluster.sh [--help] [--script script] [n:g] <arg>*"
  echo ""
  echo "--help      print this help message"
  echo "--script    run the provided script instead of DeepSpeech.py"
  echo "n           number of local nodes"
  echo "g           number of local GPUs per worker"
  echo "<arg>*      remaining parameters will be forwarded to DeepSpeech.py or a provided script"
  echo
  echo "Example usage - The following example will create a local DeepSpeech.py cluster"
  echo "with 2 nodes and 1 GPU each:"
  echo "$ run-cluster.sh 2:1 --epoch 10"
  echo
  exit 0
fi

if [[ $1 == "--script" ]]; then
  shift 1
  script=$1
  shift 1
  echo "Using script $script..."
fi

if [[ $1 =~ ([0-9]+):([0-9]+) ]]; then
  node_count=${BASH_REMATCH[1]}
  gpu_count=${BASH_REMATCH[2]}
  shift 1
fi

echo "Starting cluster with $node_count nodes and $gpu_count GPUs each..."

# Generating the node addresses
index=0
while [ "$index" -lt "$node_count" ]
do
  nodes[$index]="localhost:$((index + 3000))"
  ((index++))
done
nodes=$(printf ",%s" "${nodes[@]}")
nodes=${nodes:1}

# Starting the nodes
start=0
index=0
while [ "$index" -lt "$node_count" ]
do
    stop=$((start+gpu_count-1))
    # Creating a comma delimited number sequence from $start to $end
    cvd=`seq -s, $start $stop`
    CUDA_VISIBLE_DEVICES=$cvd $script --nodes $nodes --task_index=$index "$@" 2>&1 | sed 's/^/[node '"$index"'] /' &
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
