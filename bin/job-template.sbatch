#!/bin/bash

#SBATCH --job-name=__NAME__
#SBATCH --output=common.log
#SBATCH --ntasks=__NODES__
#SBATCH --nodes=__NODES__
#SBATCH --gres=gpu:__GPUS__

{

set -o pipefail
# force-killing all sub-processes on process exit, Ctrl-C, kill-signal
# "trap - SIGTERM" hinders "kill" from being killed itself
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

PWD=`pwd`
compute_cmd="/bin/bash .compute"
srun="srun --exclusive -N1 -n1"

# unfolding slurm's compact cluster representation
nodes_raw=`scontrol show hostname $SLURM_JOB_NODELIST`
index=0
for node in $nodes_raw
do
  # keeping nodes as array for later index lookup
  nodes[$index]=$node
  ((index=index + 1))
done
# comma separated node list (with leading comma)
raw_node_list=$(printf ",%s" "${nodes[@]}")

# exporting COMPUTE variables for the .compute script that will
# get executed on every node of the allocated cluster
export COMPUTE_NODES=${raw_node_list:1}
export COMPUTE_NODES_COUNT=${#nodes[@]}
export COMPUTE_ID="__ID__"
export COMPUTE_NAME="__NAME__"
export COMPUTE_JOB_NUMBER=$SLURM_JOB_ID
export COMPUTE_DATA_DIR=/data/shared
export COMPUTE_RESULTS_DIR=$PWD/results
export COMPUTE_KEEP_DIR=$COMPUTE_RESULTS_DIR/keep
export COMPUTE_JOB_LOG="common.log"
export COMPUTE_GLOBAL_LOG="__GLOBAL_LOG__"

for index in $(seq 0 $((COMPUTE_NODES_COUNT-1)));
do
  # will tell every instance of the .compute script, which node of the cluster it represents
  export COMPUTE_NODE_INDEX=$index
  # the node has to be specified by "-w" to guarantee execution under the correct COMPUTE_NODE_INDEX
  # if some log line contains "GLOBAL LOG", the remaining string is emitted to the provided log file
  $srun -w ${nodes[index]} /bin/bash -c "cd src && $compute_cmd" \
    | tee >(sed -n -e 's/^.*GLOBAL LOG: //p' >> $COMPUTE_GLOBAL_LOG) &
done

for index in $(seq 1 $COMPUTE_NODES_COUNT);
do
  # "wait -n" waits for any sub-process to exit
  # doing this COMPUTE_NODES_COUNT times will wait for all node sub-processes to finish
  # in case of any node sub-process failing, it will exit immediately
  wait -n
  code=$?
  if ((code > 0)); then
    echo "One node failed with exit code $code."
    exit $code
  else
    echo "One node succeeded."
  fi
done

echo "Success. Quitting..."

} 2>&1 | ts "[%Y-%m-%d %H:%M:%.S] [common  ]" > common.log
