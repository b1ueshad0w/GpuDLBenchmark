#!/usr/bin/env python
# coding=utf-8

""" benchmark.py: Execute tests.

Created by gogleyin on 3/22/18.
"""

import argparse
import os
import time
from globalconfig import FCN
from nvidiasmi import GPUAccounting

import logging
logger = logging.getLogger(__name__ if __name__ != '__main__' else os.path.splitext(os.path.basename(__file__))[0])
logger.setLevel(logging.DEBUG)

# Parse arguments, don't change input args
current_time = time.ctime()


def run(log_dir, log_file, devId, gpuCount, lr, netType,
        batchSize=64, network=FCN.fcn5, numEpochs=10,
        epochSize=50000, numThreads=8, cpuCount=1, hostFile=None, test_summary_file=None):
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    # Set system variable
    # See: https://www.tensorflow.org/performance/performance_guide
    os.environ['OMP_NUM_THREADS'] = cpuCount  # Specifies the number of threads to use.
    os.environ['OPENBLAS_NUM_THREADS'] = cpuCount
    os.environ['MKL_NUM_THREADS'] = cpuCount

    # Build cmd for benchmark
    root_path = os.path.dirname(os.path.abspath(__file__))
    tool_dir = os.path.join(root_path, netType, network)
    envs = {
        'CUDA_VISIBLE_DEVICES': devId,
    }
    args = {
        'batch_size': batchSize,
        'epochs': numEpochs,
        'epoch_size': epochSize,
        'test_summary_file': test_summary_file,
        'learning_rate': lr,
    }
    script = '%s_bm.py' % network
    if int(gpuCount) > 1:
        script = '%s_multigpu_bm.py' % network
        args['gpu_count'] = gpuCount
        args['device_ids'] = devId
    else:
        args['xla'] = True
        args['use_datasets'] = True
        args['device_id'] = devId
    script_path = os.path.join(tool_dir, script)
    if not os.path.isfile(script_path):
        logger.error('Script file not found: %s' % script_path)
    envs_str = ' '.join(['%s=%s' % (k, v) for k, v in envs.items()])
    args_str = ' '.join(['--%s=%s' % (k, v) for k, v in args.items()])
    cmd = '%s python %s %s &> %s' % (envs_str, script_path, args_str, log_file)

    gpu_usage_csv = os.path.join(log_dir, script.replace('py', 'csv'))

    # Add time info into the log
    start_time = time.time()
    logger.debug('Executing shell: %s' % cmd)
    with GPUAccounting(gpu_usage_csv):
        if os.system(cmd) != 0:
            logger.error('Executing shell failed: %s.' % cmd)
        else:
            logger.debug('Executing shell success: %s' % cmd)
    time_elapsed = time.time() - start_time
    with open(log_file, "a") as logFile:
        logFile.write("Total time: " + str(time_elapsed) + "\n")
        logFile.write("cmd: " + cmd + "\n")


def set_launch_args():
    parser = argparse.ArgumentParser(description='Python script benchmarking mxnet')
    parser.add_argument('-log', type=str, default=('mxnet_' + current_time + '.log').replace(" ", "_"),
                        help='Name of log file, default= mxnet_ + current time + .log')
    parser.add_argument('-batchSize', type=int, default=64, help='Batch size for each GPU, default = 64')
    parser.add_argument('-network', type=str, default='fcn5',
                        help='name of network[fcn5 | alexnet | resnet | lstm | lstm32 | lstm64]')
    parser.add_argument('-devId', type=str, help='CPU: -1, GPU:0,1,2,3(Multiple gpu supported)')
    parser.add_argument('-numEpochs', type=int, default=10, help='number of epochs, default=10')
    parser.add_argument('-epochSize', type=int, default=50000, help='number of training data per epoch')
    parser.add_argument('-numThreads', type=int, default=8, help='number of Threads, default=8')
    parser.add_argument('-hostFile', type=str,
                        help='path to running hosts(config in host file) for multiple machine training.')
    parser.add_argument('-gpuCount', type=int, help='number of gpus in used')
    parser.add_argument('-cpuCount', type=int, default=1, help='number of cpus in used for cpu version')
    parser.add_argument('-lr', type=str, help='learning rate')
    parser.add_argument('-netType', type=str, help='network type')
    parser.add_argument('-log_dir', type=str, help='Directory to save logs.')
    parser.add_argument('-test_summary_file', type=str, help='File to record benchmark result.')
    args = parser.parse_args()
    # print(args)
    run(log_file=args.log, batchSize=args.batchSize, network=args.network, devId=args.devId, numEpochs=args.numEpochs,
        epochSize=args.epochSize, numThreads=args.numThreads, hostFile=args.hostFile, gpuCount=args.gpuCount,
        cpuCount=args.cpuCount, lr=args.lr, netType=args.netType)


if __name__ == '__main__':
    set_launch_args()