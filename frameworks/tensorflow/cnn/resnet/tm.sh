#!/usr/bin/env bash

start=`date +%s.%N`
CUDA_VISIBLE_DEVICES=$deviceId python ${script_path} --train_dir=${train_dir} --batch_size=$batch_size --epochs=$epochs --epoch_size=${epoch_size} --device_ids=$deviceId --num_gpus=${gpu_count} &> $logFile
end=`date +%s.%N`
runtime=$( echo "$end - $start" | bc -l )
echo "finished with execute time: ${runtime}"
