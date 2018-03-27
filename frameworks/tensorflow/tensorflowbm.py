#!/usr/bin/env python
# coding=utf-8

""" benchmark.py: Execute tests.

Created by gogleyin on 3/22/18.
"""

import argparse
import re
import os
import time
import shutil
import subprocess
from globalconfig import FCN, CNN, RNN
from nvidiasmi import GPUAccounting
from benchmark import TestConfigEntry, Framework, NetworkType, Status, Synthetic, TestResultEntry
from extract_info import extract_info_tensorflow, extract_info_tensorflow_synthetic
from frameworks.tensorflow.cnn.resnet.resnet_bm import EPOCH_SIZE as RESNET_EPOCH_SIZE
from frameworks.tensorflow.cnn.alexnet.alexnet_bm import EPOCH_SIZE as ALEXNET_EPOCH_SIZE
from frameworks.tensorflow.fc.fcn5.fcn5_mnist import EPOCH_SIZE as FCN_EPOCH_SIZE
import logging
logger = logging.getLogger(__name__ if __name__ != '__main__' else os.path.splitext(os.path.basename(__file__))[0])
logger.setLevel(logging.DEBUG)

# Parse arguments, don't change input args
current_time = time.ctime()


def get_script(gpu_count, network, tool_dir, synthetic=False):
    if synthetic:
        synthetic_dir = os.path.join(tool_dir, 'synthetic')
        if not os.path.isdir(synthetic_dir):
            logger.warning('Synthetic directory not found: %s.' % (synthetic_dir,))
            return
        synthetic_script = os.path.join(synthetic_dir, '%s_synthetic.py' % network)
        if not os.path.isfile(synthetic_script):
            logger.warning('Synthetic script not found: %s' % (synthetic_script,))
            return
        return synthetic_script
    script_name = '%s_%sbm.py' % (network, 'multigpu_' if gpu_count > 1 else '')
    script = os.path.join(tool_dir, script_name)
    if not os.path.isfile(script):
        logger.warning('Script not found: %s' %(script,))
        return
    return script


def run(log_dir,
        log_file,
        devId,
        gpuCount,
        lr,
        netType,
        batchSize=64,
        network=FCN.fcn5,
        numEpochs=10,
        epochSize=50000,
        numThreads=8,
        cpuCount=1,
        hostFile=None,
        test_summary_file=None,
        synthetic=Synthetic.false):
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
        'learning_rate': lr,
        'training_summary_file': os.path.join(log_dir, 'training_summary.log')
    }
    if int(gpuCount) > 1:
        args['gpu_count'] = gpuCount
        args['device_ids'] = devId
    else:
        args['xla'] = True
        args['use_datasets'] = True
        args['device_id'] = devId
    script_path = get_script(gpuCount, network, tool_dir, synthetic)
    if not script_path or not os.path.isfile(script_path):
        logger.error('Script file not found: %s' % script_path)
        return

    envs_str = ' '.join(['%s=%s' % (k, v) for k, v in envs.items()])
    args_str = ' '.join(['--%s=%s' % (k, v) for k, v in args.items()])
    cmd = '%s python %s %s &> %s' % (envs_str, script_path, args_str, log_file)

    script_name = os.path.basename(script_path)
    gpu_usage_csv = os.path.join(log_dir, script_name.replace('py', 'csv'))

    # Add time info into the log
    start_time = time.time()
    logger.debug('Executing shell: %s' % cmd)
    with GPUAccounting(gpu_usage_csv):
        if os.system(cmd) != 0:
            logger.error('Executing shell failed: %s.' % cmd)
            return
        logger.debug('Executing shell success: %s' % cmd)
    time_elapsed = time.time() - start_time
    with open(log_file, "a") as logFile:
        logFile.write("Total time: " + str(time_elapsed) + "\n")
        logFile.write("cmd: " + cmd + "\n")
    # if test_summary_file:
    #     with open(test_summary_file, 'a') as f:
    #         config = TestConfigEntry(Framework.tensorflow, NetworkType.fc, FCN.fcn5, 0, 1, 4096, 2, 60000, 0.05, Status.enabled)
    #         abc = ','.join([str(i) for i in config])


def run_synthetic():
    pass


def save_benchmark_result(benchmark_training_speed, benchmark_accuracy):
    pass


def evaluation_cnn(tool_path, log_dir, train_dir, train_log_path):
    eval_script = os.path.join(tool_path, 'eval.sh')
    if not os.path.isfile(eval_script):
        logger.warning('Script for evaluation not found: %s' % eval_script)
        return
    eval_log_path = os.path.join(log_dir, 'evaluation.log')
    eval_dir = os.path.join(log_dir, 'eval_dir_%s' % str(int(time.time())))
    if os.path.isdir(eval_dir):
        shutil.rmtree(eval_dir)
    eval_envs = {
        'checkpoint_dir': train_dir,
        'train_log': train_log_path,
        'eval_log': eval_log_path,
        'eval_dir': eval_dir,
    }
    eval_envs_str = ' '.join(['%s=%s' % (k, v) for k, v in eval_envs.items()])
    eval_cmd = '%s bash %s &> %s' % (eval_envs_str, eval_script, eval_log_path)
    logger.debug('Begin evaluation.')
    logger.debug('Executing shell: %s' % eval_cmd)
    if os.system(eval_cmd) != 0:
        logger.error('Executing shell failed: %s' % eval_cmd)
    else:
        logger.debug('Executing shell success: %s' % eval_cmd)
    shutil.rmtree(eval_dir)

    with open(eval_log_path, 'r') as f:
        content = f.read()
        pattern = 'precision @ 1 = (\d+\.\d+)'
        result = re.search(pattern, content)
        if not result:
            logging.error('Could not find accuracy info in cnn train log: %s' % train_log_path)
            return 'err'
        return result.group(1)


def evaluation_fcn5(train_log_path):
    with open(train_log_path, 'r') as f:
        content = f.read()
        pattern = 'test accuracy (\d+\.\d+)\naverage_batch_time'
        result = re.search(pattern, content)
        if not result:
            logging.error('Could not find accuracy info in fc5 train log: %s' % train_log_path)
            return 'err'
        return result.group(1)


def evaluation_rnn(train_log_path):
    with open(train_log_path, 'r') as f:
        content = f.read()
        pattern = 'Test Perplexity: (\d+\.\d+)'
        result = re.search(pattern, content)
        if not result:
            logging.error('Could not find accuracy info in rnn train log: %s' % train_log_path)
            return 'err'
        return result.group(1)


def evaluation(tool_path, log_dir, train_dir, train_log_path):
    network_name = os.path.basename(tool_path)
    if network_name == CNN.resnet or network_name == CNN.alexnet:
        return evaluation_cnn(tool_path, log_dir, train_dir, train_log_path)
    elif network_name == FCN.fcn5:
        return evaluation_fcn5(train_log_path)
    elif network_name == RNN.lstm:
        return evaluation_rnn(train_log_path)
    return 'err'


def get_epoch_size(network):
    default = 50000
    if network == CNN.alexnet:
        return ALEXNET_EPOCH_SIZE
    if network == CNN.resnet:
        return RESNET_EPOCH_SIZE
    if network == FCN.fcn5:
        return FCN_EPOCH_SIZE
    logger.warning('Not setting epoch size! Will set to default: %s' % default)
    return default


def run_sh(log_dir, dev_id, net_type, network, gpu_count, learning_rate, cpu_count=1, batch_size=64, num_epochs=10,
           epoch_size=None, synthetic=Synthetic.false, test_result_file=None):
    benchmark_training_speed, benchmark_accuracy = '-', '-'
    log_path = os.path.join(log_dir, 'training.log')
    train_dir = os.path.join(log_dir, 'train-dir-%s' % str(int(time.time())))
    gpu_usage_csv = os.path.join(log_dir, 'gpu-accounting.csv')

    if os.path.isdir(train_dir):
        shutil.rmtree(train_dir)

    if not epoch_size:
        epoch_size = get_epoch_size(network)

    # Set system variable
    os.environ['OMP_NUM_THREADS'] = cpu_count
    os.environ['OPENBLAS_NUM_THREADS'] = cpu_count
    os.environ['MKL_NUM_THREADS'] = cpu_count

    # Build cmd for benchmark
    root_path = os.path.dirname(os.path.abspath(__file__))
    tool_path = os.path.join(root_path, net_type, network)
    if not os.path.isdir(tool_path):
        logging.warning('Network not found: %s' % (tool_path,))
        save_benchmark_result(benchmark_training_speed, benchmark_accuracy)
        return

    envs = {
        'CUDA_VISIBLE_DEVICE': dev_id,
        'deviceId': dev_id,
        'batch_size': batch_size,
        'epochs': num_epochs,
        'epoch_size': epoch_size,
        'train_dir': train_dir,
        'learning_rate': learning_rate,
    }
    script_name = 't.sh'
    envs['script_path'] = os.path.join(tool_path, '%s_bm.py' % network)
    if int(gpu_count) > 1:
        envs['gpu_count'] = gpu_count
        envs['script_path'] = os.path.join(tool_path, '%s_multigpu_bm.py' % network)
        script_name = 'tm.sh'
    if synthetic:
        script_name = os.path.join('synthetic',  't.sh')
        envs['script_path'] = os.path.join(tool_path, 'synthetic', '%s_synthetic.py' % network)
    script_path = os.path.join(tool_path, script_name)
    if not os.path.isfile(script_path):
        logging.warning('Script not found: %s' % (script_path,))
        save_benchmark_result(benchmark_training_speed, benchmark_accuracy)
        return

    envs_str = ' '.join(['%s=%s' % (k, v) for k, v in envs.items()])
    cmd = '%s bash %s &> %s' % (envs_str, script_path, log_path)

    start_time = time.time()
    logger.debug('Executing shell: %s' % cmd)
    with GPUAccounting(gpu_usage_csv):
        if os.system(cmd) != 0:
            logger.error('Executing shell failed: %s.' % cmd)
            save_benchmark_result(benchmark_training_speed, benchmark_accuracy)
            return
        logger.debug('Executing shell success: %s' % cmd)
    time_elapsed = time.time() - start_time

    # Parse log file and extract benchmark info
    # average_batch_time
    benchmark_training_speed = extract_info_tensorflow_synthetic(log_path) \
        if synthetic else extract_info_tensorflow(log_path)

    # Evaluation
    if not synthetic:
        benchmark_accuracy = evaluation(tool_path, log_dir, train_dir, log_path)

    # Save log file
    with open(log_path, "a") as logFile:
        logFile.write("\nTotal time: %s\ncmd: %s" % (str(time_elapsed), cmd))

    test_result = TestResultEntry(Framework.tensorflow, net_type, network, dev_id, gpu_count, batch_size, num_epochs,
                                  epoch_size, learning_rate, synthetic, benchmark_training_speed, benchmark_accuracy)

    if test_result_file and os.path.isfile(test_result_file):
        with open(test_result_file, 'a') as f:
            f.write('\n%s' % test_result)

    shutil.rmtree(train_dir)


def set_launch_args():
    parser = argparse.ArgumentParser(description='Python script benchmarking mxnet')
    parser.add_argument('-log_dir', type=str, help='Directory for logs.')
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
    parser.add_argument('-synthetic', type=bool, default=False, help='whether to use the synthetic data')
    args = parser.parse_args()
    # print(args)
    run(log_dir=args.log_dir,
        log_file=args.log,
        devId=args.devId,
        gpuCount=args.gpuCount,
        lr=args.lr,
        netType=args.netType,
        batchSize=args.batchSize,
        network=args.network,
        numEpochs=args.numEpochs,
        epochSize=args.epochSize,
        numThreads=args.numThreads,
        cpuCount=args.cpuCount,
        hostFile=args.hostFile,
        test_summary_file=args.test_summary_file,
        synthetic=args.synthetic,
        )


if __name__ == '__main__':
    set_launch_args()