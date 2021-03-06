#!/usr/bin/env bash

start=`date +%s.%N`
# Please be noted that we now update TF to the version of 1.2, and the last two parameters can work. Thanks to tfboyd
CUDA_VISIBLE_DEVICES=$deviceId python ${script_path} \
 --train_dir=${train_dir} \
 --batch_size=$batch_size \
 --epochs=$epochs \
 --epoch_size=${epoch_size} \
 --learning_rate=${learning_rate} \
 --device_id=$deviceId \
 &> $logFile
end=`date +%s.%N`
runtime=$( echo "$end - $start" | bc -l )
echo "finished with execute time: ${runtime}"
