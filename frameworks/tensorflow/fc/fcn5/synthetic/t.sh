#!/usr/bin/env bash

python ${script_path} --epochs=${epochs} --epoch_size=${epoch_size} --minibatch=${batch_size} --deviceid=${deviceId} &> ${logFile}