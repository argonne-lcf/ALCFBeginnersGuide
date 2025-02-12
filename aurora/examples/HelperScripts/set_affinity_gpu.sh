#!/usr/bin/env bash

num_gpu=6
num_tile=2

gpu_id=$(( (PALS_LOCAL_RANKID / num_tile ) % num_gpu ))
tile_id=$((PALS_LOCAL_RANKID % num_tile))

unset EnableWalkerPartition
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
export ZE_AFFINITY_MASK=$gpu_id.$tile_id
ulimit -c 0 # Disabled for time being

#echo “RANK= ${PALS_RANKID} LOCAL_RANK= ${PALS_LOCAL_RANKID} gpu= ${gpu_id}  tile= ${tile_id}”

#https://stackoverflow.com/a/28099707/7674852
"$@"

