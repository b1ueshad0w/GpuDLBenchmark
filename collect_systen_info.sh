#!/usr/bin/env bash


CollectEnv()
{
  vars_count=$#
  expected_count=1
  if (( $vars_count < $expected_count )); then
    echo "Error: Function CollectEnv needs at least $expected_count argument(s)!"
    exit 1
  fi

  logFile=$1
  echo "============= Querying driver configuations: =============" >> $logFile
  nvidia-smi -q >> $logFile

  echo "============= Querying CUDA version: =====================" >> $logFile
  nvcc --version >> $logFile

  echo "============= Querying cuDNN version: ====================" >> $logFile
  cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2 >> $logFile

  echo "============= Querying Tensorflow version: ===============" >> $logFile
  # source ~/envs/tensorflow/bin/activate
  pip show tensorflow-gpu >> $logFile
  # deactivate

  echo "============= Querying Linux version: ====================" >> $logFile
  cat /proc/version >> $logFile

  echo "============= Querying CPU info: =========================" >> $logFile
  cat /proc/cpuinfo >> $logFile

  echo "============= Querying Memory info: ======================" >> $logFile
  cat /proc/meminfo >> $logFile
}


CollectEnv ${logPath}