#!/usr/bin/env bash

python ${script_path} --batchSize=${batch_size} --eval_dir=${eval_dir} --checkpoint_dir=${checkpoint_dir} &> ${logFile}
