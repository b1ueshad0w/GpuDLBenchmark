#!/usr/bin/env bash

start=`date +%s.%N`
CUDA_VISIBLE_DEVICES=$deviceId python ${script_path} --batchsize=$batch_size --max_max_epoch=$epochs --device=$deviceId &> $logFile
end=`date +%s.%N`
runtime=$( echo "$end - $start" | bc -l )
echo "finished with execute time: ${runtime}"
