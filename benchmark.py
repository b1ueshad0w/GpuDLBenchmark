#!/usr/bin/env python
# coding=utf-8

""" benchmark.py: Execute tests according to a given config file.

Created by gogleyin on 3/22/18.
"""

from collections import namedtuple
import csv
import os
import time
import subprocess
import logging
logger = logging.getLogger(__name__ if __name__ != '__main__' else os.path.splitext(os.path.basename(__file__))[0])
logger.setLevel(logging.DEBUG)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
HOST_NAME = subprocess.check_output("hostname", shell=True).strip().split('\n')[0]

FIELDS = [
    'framework',
    'network_type',
    'network_name',
    'device_id',
    'device_count',
    'batch_size',
    'number_of_epochs',
    'epoch_size',
    'learning_rate',
    'enabled'
]

TestConfigEntry = namedtuple('TestConfigEntry', FIELDS)


class Framework(object):
    tensorflow = 'tensorflow'


class FCN(object):
    fcn5 = 'fcn5'


class CNN(object):
    alexnet = 'alexnet'
    resnet = 'resnet'


class RNN(object):
    lstm = 'lstm'


class NetworkType(object):
    fc = 'fc'
    cnn = 'cnn'
    rnn = 'rnn'


class Status(object):
    enabled = '1'
    disabled = '0'


def generate_configs(config_file):
    config = TestConfigEntry(Framework.tensorflow, NetworkType.fc, FCN.fcn5, 0, 1, 4096, 2, 60000, 0.05, Status.enabled)
    with open(config_file, 'wb') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(FIELDS)
        writer.writerow(config)


def generate_log_file(config):
    if "-1" in config.device_id:
        device_name = 'cpuName'
    else:
        device_name = 'gpuName'
    log_file_name = '-'.join([config.framework, config.network_type, config.network_name, device_name,
                              '*%s' % config.device_count,
                              'b%s' % config.batch_size,
                             time.ctime(), HOST_NAME + '.log'])
    log_file_name = log_file_name.replace(" ", "_")
    return log_file_name


def run(config_file):
    with open(config_file, 'rb') as csv_file:
        reader = csv.reader(csv_file)
        next(reader, None)  # skip the header
        configs = [TestConfigEntry(*row) for row in reader]
        for config in configs:
            if config.enabled != Status.enabled:
                continue
            logger.info('Running test with config: %s' % config)
            sub_benchmark_file_name = config.framework + 'bm.py'
            sub_benchmark = os.path.join(PROJECT_ROOT, 'frameworks', config.framework, sub_benchmark_file_name)
            if not os.path.exists(sub_benchmark):
                logger.error('File not found: %s' % (sub_benchmark,))
                continue
            log_file_name = generate_log_file(config)
            log_file_path = log_file_name
            args = {
                'netType': config.network_type,
                'log': log_file_path,
                'batchSize': config.batch_size,
                'network': config.network_name,
                'lr': config.learning_rate
            }
            args_str = ' '.join(['-%s %s' % (k, v) for k, v in args.items()])
            cmd = 'python {scriptFile} {argsStr}'.format(scriptFile=sub_benchmark, argsStr=args_str)
            logger.debug('Executing shell: %s' % cmd)
            subprocess.check_call(cmd, shell=True)


if __name__ == '__main__':
    _config_file = '/tmp/config.csv'
    # generate_configs(_config_file)
    run(_config_file)
