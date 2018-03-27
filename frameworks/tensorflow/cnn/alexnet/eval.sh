#!/usr/bin/env bash

python alexnet_eval.py --eval_dir=${eval_dir} --checkpoint_dir=${checkpoint_dir} &> ${eval_log}