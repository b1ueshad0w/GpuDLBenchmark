#!/usr/bin/env bash

start=`date +%s.%N`
CUDA_VISIBLE_DEVICES=$deviceId python ${script_path} --train_dir=${train_dir} --batch_size=$batch_size --epochs=$epochs --device_ids=$deviceId --num_gpus=${gpu_count}
end=`date +%s.%N`
runtime=$( echo "$end - $start" | bc -l )
echo "finished with execute time: ${runtime}"
