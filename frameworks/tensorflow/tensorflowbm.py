#!/usr/bin/env python
# coding=utf-8

""" benchmark.py: Execute tests.

Created by gogleyin on 3/22/18.
"""

import argparse
import os
import time

import logging
logger = logging.getLogger(__name__ if __name__ != '__main__' else os.path.splitext(os.path.basename(__file__))[0])
logger.setLevel(logging.DEBUG)

# Parse arguments, don't change input args
current_time = time.ctime()


def run(log_file, batchSize, network, devId,
        numEpochs, epochSize, numThreads, hostFile,
        gpuCount, cpuCount, lr, netType):
    # Set system variable
    os.environ['OMP_NUM_THREADS'] = cpuCount
    os.environ['OPENBLAS_NUM_THREADS'] = cpuCount
    os.environ['MKL_NUM_THREADS'] = cpuCount

    # Build cmd for benchmark
    root_path = os.path.dirname(os.path.abspath(__file__))
    tool_path = os.path.join(root_path, netType, network)
    if not os.path.exists(tool_path)
    if ".log" not in log_file:
        log_file += ".log"
    log_path = os.getcwd() + "/" + log_file
    envs = {
        'CUDA_VISIBLE_DEVICE': devId,
        'deviceId': devId,
        'batch_size': batchSize,
        'epochs': numEpochs,
    }
    script = 't.sh'
    if int(devId) >= 0 and int(gpuCount) > 1:
        script = 'tm.sh'
        envs['gpu_count'] = gpuCount
    script_path = os.path.join(tool_path, script)
    envs_list = ['%s=%s'%(k,v) for k, v in envs.items()]
    envs_str = ' '.join(envs_list)
    cmd = '{envStr} {scriptPath} >& {logPath}'.format(envStr=envs_str, scriptPath=script_path, logPath=log_path)

    # Add time info into the log
    start_time = time.time()
    logger.info('Executing shell: %s' % cmd)
    os.system(cmd)
    time_elapsed = time.time() - start_time
    with open(log_path, "a") as logFile:
        logFile.write("Total time: " + str(time_elapsed) + "\n")
        logFile.write("cmd: " + cmd + "\n")


def set_launch_args():
    parser = argparse.ArgumentParser(description='Python script benchmarking mxnet')
    parser.add_argument('-log', type=str, default=('mxnet_' + current_time + '.log').replace(" ", "_"),
                        help='Name of log file, default= mxnet_ + current time + .log')
    parser.add_argument('-batchSize', type=str, default='64', help='Batch size for each GPU, default = 64')
    parser.add_argument('-network', type=str, default='fcn5',
                        help='name of network[fcn5 | alexnet | resnet | lstm | lstm32 | lstm64]')
    parser.add_argument('-devId', type=str, help='CPU: -1, GPU:0,1,2,3(Multiple gpu supported)')
    parser.add_argument('-numEpochs', type=str, default='10', help='number of epochs, default=10')
    parser.add_argument('-epochSize', type=str, default='50000', help='number of training data per epoch')
    parser.add_argument('-numThreads', type=str, default='8', help='number of Threads, default=8')
    parser.add_argument('-hostFile', type=str,
                        help='path to running hosts(config in host file) for multiple machine training.')
    parser.add_argument('-gpuCount', type=str, help='number of gpus in used')
    parser.add_argument('-cpuCount', type=str, default='1', help='number of cpus in used for cpu version')
    parser.add_argument('-lr', type=str, help='learning rate')
    parser.add_argument('-netType', type=str, help='network type')
    args = parser.parse_args()
    # print(args)
    run(log_file=args.log, batchSize=args.batchSize, network=args.network, devId=args.devId, numEpochs=args.numEpochs,
        epochSize=args.epochSize, numThreads=args.numThreads, hostFile=args.hostFile, gpuCount=args.gpuCount,
        cpuCount=args.cpuCount, lr=args.lr, netType=args.netType)


if __name__ == '__main__':
    set_launch_args()