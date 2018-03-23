#!/usr/bin/env python
# coding=utf-8

""" nvidiasmi.py: A python wrapper for nvidia-smi command line tool.

Created by gogleyin on 3/22/18.
"""


import subprocess
import re

TOOL = 'sudo nvidia-smi'


class ModeStatus(object):
    On = True
    Off = False


class GPUMode(object):
    def __init__(self, device_id):
        self._device_id = str(device_id)
        self._cmd_prefix = TOOL
        # if '-i' option is not provided, the cmd will effect all GPUs.
        if device_id and len(device_id) > 0:
            self._cmd_prefix = TOOL + ' -i %s' % device_id

    @property
    def status(self):
        return True

    def turn_on(self):
        pass

    def turn_off(self):
        pass


class EccMode(GPUMode):
    def __init__(self, device_id):
        super(EccMode, self).__init__(device_id)

    @property
    def status(self):
        cmd = '%s --query-gpu=ecc.mode.current --format=csv' % (self._cmd_prefix,)
        return 'Enabled' in subprocess.check_output(cmd, shell=True)

    def turn_on(self):
        cmd = '%s -e 1' % (self._cmd_prefix,)
        subprocess.check_call(cmd)
        raise RuntimeError('Reboot is required for ECC mode changes.')

    def turn_off(self):
        cmd = '%s -e 0' % (self._cmd_prefix,)
        subprocess.check_call(cmd)
        raise RuntimeError('Reboot is required for ECC mode changes.')


class PersistenceMode(GPUMode):
    def __init__(self, device_id):
        super(PersistenceMode, self).__init__(device_id)

    @property
    def status(self):
        cmd = '%s --query-gpu=persistence_mode --format=csv' % (self._cmd_prefix,)
        return 'Enabled' in subprocess.check_output(cmd, shell=True)

    def turn_on(self):
        cmd = '%s -pm 1' % (self._cmd_prefix,)
        subprocess.check_call(cmd)

    def turn_off(self):
        cmd = '%s -pm 0' % (self._cmd_prefix,)
        subprocess.check_call(cmd)


class AccountingMode(GPUMode):
    def __init__(self, device_id):
        super(AccountingMode, self).__init__(device_id)

    @property
    def status(self):
        cmd = '%s --query-gpu=accounting.mode --format=csv' % (self._cmd_prefix,)
        return 'Enabled' in subprocess.check_output(cmd, shell=True)

    def turn_on(self):
        cmd = '%s -am 1' % (self._cmd_prefix,)
        subprocess.check_call(cmd)

    def turn_off(self):
        cmd = '%s -am 0' % (self._cmd_prefix,)
        subprocess.check_call(cmd)


class AutoBoostMode(GPUMode):
    def __init__(self, device_id):
        super(AutoBoostMode, self).__init__(device_id)

    @property
    def status(self):
        cmd = '%s -i 0 -q -d CLOCK'
        output = subprocess.check_output(cmd, shell=True)
        return 'On' in output.splitlines()[-3]

    def turn_on(self):
        cmd = '%s --auto-boost-default=DISABLED' % (self._cmd_prefix,)
        subprocess.check_call(cmd)

    def turn_off(self):
        cmd = '%s --auto-boost-default=ENABLED' % (self._cmd_prefix,)
        subprocess.check_call(cmd)


class GPU(object):
    def __init__(self, device_id=None):
        self._device_id = device_id
        self._cmd_prefix = TOOL
        # if '-i' option is not provided, the cmd will effect all GPUs.
        if device_id and len(device_id) > 0:
            self._cmd_prefix = TOOL + ' -i %s' % device_id

        self.persistence_mode = PersistenceMode(device_id)
        self.ecc_mode = EccMode(device_id)
        self.accounting_mode = AccountingMode(device_id)
        self.auto_boost_mode = AutoBoostMode(device_id)

    def get_accounting_info(self, to_file=None):
        cmd = '%s --query-accounted-apps=timestamp,pid,time,gpu_name,gpu_util,mem_util,max_memory_usage ' \
              '--format=csv' % (self._cmd_prefix,)
        if to_file:
            cmd += ' > %s' % to_file
        subprocess.check_call(cmd, shell=True)


class GPUManager(object):
    def __init__(self):
        pass

    @classmethod
    def list_gpus(cls):
        cmd = 'TOOL -q'
        output = subprocess.check_output(cmd)
        pattern = 'Attached GPUs\s*:\s*(\d+)'
        result = re.search(pattern, output)
        gpu_count = int(result.group(1))
        return [GPU(str(i)) for i in range(gpu_count)]


class GPUAccounting(object):
    def __init__(self, log_to_file, device_id=None):
        self._log_file_path = log_to_file
        self._gpu = GPU(device_id)

    def __enter__(self):
        self._gpu.accounting_mode.turn_off()
        self._gpu.accounting_mode.turn_on()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._gpu.get_accounting_info(self._log_file_path)


if __name__ == '__main__':
    gpu = GPU()
    gpu.ecc_mode.turn_off()
