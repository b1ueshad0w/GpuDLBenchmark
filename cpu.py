#!/usr/bin/env python
# coding=utf-8

""" cpu.py: Manipulate CPU settings.

Created by gogleyin on 4/23/18.
"""

import os
import re
import logging
logger = logging.getLogger(__name__ if __name__ != '__main__' else os.path.splitext(os.path.basename(__file__))[0])
logger.setLevel(logging.DEBUG)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
cpu_sh = os.path.join(PROJECT_ROOT, 'cpu.sh')
CPU_DIR = '/sys/devices/system/cpu'
PROC_CPU_FILE = '/proc/cpuinfo'


class CPUStatus(object):
    On = '1'
    Off = '0'
    validStatus = [On, Off]


def turn_on_cpu(cpu_id):
    set_cpu_status(cpu_id, CPUStatus.On)


def turn_off_cpu(cpu_id):
    set_cpu_status(cpu_id, CPUStatus.Off)


def set_cpu_status(cpu_id, status):
    """
    Toggle CPU states.
    Issue: It is weird that we cannot directly modify online file content, which results in a PermissionError, even
    with SUDO. The only way is by calling 'sudo bash bashScript.sh' and move the logic (changing the file content) into
    the bashScript.sh .
    shell script.
    :param cpu_id: like "cpu0", "cpu1"
    :param status: 0 or 1
    :return: None
    """
    if cpu_id == 'cpu0':
        logger.debug('cpu0 cannot be modified.')
        return
    if status not in CPUStatus.validStatus:
        raise RuntimeError('Invalid CPU status: %s. Must belongs to: %s' % (status, CPUStatus.validStatus))
    online_file_path = os.path.join(CPU_DIR, cpu_id, 'online')
    cmd = 'sudo bash %s %s %s' % (cpu_sh, status, online_file_path)
    ret = os.system(cmd)
    if ret != 0:
        logger.error('shell execution failed: %s' % cmd)
        exit(1)


def get_cpu_status(cpu_id):
    if cpu_id == 'cpu0':
        return CPUStatus.On
    online_file_path = os.path.join(CPU_DIR, cpu_id, 'online')
    with open(online_file_path, 'rb') as f:
        status = f.read().strip()
        if status not in CPUStatus.validStatus:
            raise RuntimeError('Unexpected status: %s' % status)
        return status


def get_cpu_count_via_cpuinfo():
    with open(PROC_CPU_FILE, 'rb') as cpu_file:
        content = cpu_file.read()
        return len(content.split('\n\n')) - 1


class CPU(object):
    def __init__(self, cpu_id):
        self._id = cpu_id

    def turn_on(self):
        turn_on_cpu(self._id)

    def turn_off(self):
        turn_off_cpu(self._id)

    def set_status(self, status):
        if status not in CPUStatus.validStatus:
            raise RuntimeError('Invalid CPU status: %s. Must belongs to: %s' % (status, CPUStatus.validStatus))
        set_cpu_status(self._id, status)

    @property
    def status(self):
        return get_cpu_status(self._id)


class CPUManager(object):
    def __init__(self):
        pass

    @staticmethod
    def get_all_cpus():
        # todo We should get all cpus via '/proc/cpuinfo'
        items = os.listdir(CPU_DIR)
        cpu_id_pattern = 'cpu\d+'
        cpus = [CPU(e) for e in items if re.match(cpu_id_pattern, e) is not None]
        return cpus


ALL_CPUS = CPUManager.get_all_cpus()
ALL_CPU_COUNT = len(ALL_CPUS)


class CpuLimiter(object):
    def __init__(self, cpu_count):
        if cpu_count < 1:
            raise RuntimeError('CpuCount could not be less than 1!')
        if cpu_count > ALL_CPU_COUNT:
            raise RuntimeError('CpuCount(%s) could not be larger than ALL_CPU_COUNT(%s)' % (cpu_count, ALL_CPU_COUNT))
        self._cpu_count = cpu_count
        self._origin_states = [e.status for e in ALL_CPUS]

    def __enter__(self):
        [cpu.turn_off() for cpu in ALL_CPUS]
        [cpu.turn_on() for cpu in ALL_CPUS[:self._cpu_count]]
        logger.debug('Current enabled CPU count: %s' % get_cpu_count_via_cpuinfo())

    def __exit__(self, exc_type, exc_val, exc_tb):
        for i, cpu in enumerate(ALL_CPUS):
            cpu.set_status(self._origin_states[i])


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    cpu_count = 5
    origin_cpu_count = get_cpu_count_via_cpuinfo()
    print 'original cpu count: %s' % origin_cpu_count
    with CpuLimiter(cpu_count):
        current_cpu_count = get_cpu_count_via_cpuinfo()
        print 'current cpu count: %s' % current_cpu_count
        assert cpu_count == current_cpu_count
    final_cpu_count = get_cpu_count_via_cpuinfo()
    print 'final cpu count: %s' % final_cpu_count
    assert origin_cpu_count == final_cpu_count
