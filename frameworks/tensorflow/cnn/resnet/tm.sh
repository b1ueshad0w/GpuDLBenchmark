#!/usr/bin/env bash

real_batch_size=`expr ${batch_size} / ${gpu_count}`

start=`date +%s.%N`
CUDA_VISIBLE_DEVICES=$deviceId python ${script_path} \
    --train_dir=${train_dir} \
    --batch_size=${real_batch_size} \
    --epochs=$epochs \
    --epoch_size=${epoch_size} \
    --learning_rate=${learning_rate} \
    --device_ids=$deviceId \
    --num_gpus=${gpu_count} \
    &> $logFile
end=`date +%s.%N`
runtime=$( echo "$end - $start" | bc -l )
echo "finished with execute time: ${runtime}"
