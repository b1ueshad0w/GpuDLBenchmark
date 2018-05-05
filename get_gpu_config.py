#!/usr/bin/env python
# coding=utf-8

""" get_gpu_config.py: Get gpu config.

Created by gogleyin on 5/5/18.
"""

import xml.etree.ElementTree as ET
from xml.etree.ElementTree import tostring


children = 'children'
black_tags = {
    children: [
        'timestamp',
    ],
    'gpu': {
        children: [
            'serial',
            'uuid',
            'board_id',
            'gpu_part_number',
            'fan_speed',
            'performance_state',
            'clocks_throttle_reasons',
            'utilization',
            'encoder_stats',
            'ecc_errors',
            'retired_pages',
            'processes',
            'accounted_processes',
        ],
        'pci': [
            'pci_bus',
            'pci_device',
            'pci_domain',
            'pci_device_id',
            'pci_bus_id',
            'pci_sub_system_id',
            'replay_counter',
            'tx_util', 'rx_util',
        ],
        'fb_memory_usage': [
            'used',
            'free',
        ],
        'bar1_memory_usage': [
            'used',
            'free',
        ],
        'ecc_mode': [
            'pending_ecc',
        ],
        'temperature': [
            'gpu_temp',
            'memory_temp',
        ],
        'power_readings': [
            'power_state',
            'power_draw',
        ]
    }
}


def process_gpu_xml_file(xml_file):
    """
    Remove variable info (temperature, bud_id, etc.) in gpu info obtained from `nvidia-smi -q -x`
    For example, <gpu id="00000000:00:06.0"> will become <gpu>
    :param xml_file: xml_file whose content is obtained from `nvidia-smi -q -x`
    :return: string
    """
    tree = ET.parse(xml_file)
    return _process(tree)


def process_gpu_xml_info(xml_content):
    """
    Remove variable info (temperature, bud_id, etc.) in gpu info obtained from `nvidia-smi -q -x`
    For example, <gpu id="00000000:00:06.0"> will become <gpu>
    :param xml_content: obtained from `nvidia-smi -q -x`
    :return: string
    """
    tree = ET.fromstring(xml_content)
    _process(tree)


def _process(tree):
    root = tree.getroot()

    for child in root:
        if child.tag in black_tags[children]:
            root.remove(child)

    for gpu in root.findall('gpu'):
        gpu.attrib = {}  # remove gpu's <id> property
        for k, v in black_tags['gpu'].items():
            if k == children:
                for item in v:
                    child = gpu.find(item)
                    gpu.remove(child)
            else:
                child = gpu.find(k)
                for item in v:
                    grand_child = child.find(item)
                    child.remove(grand_child)
    return tostring(root)


if __name__ == '__main__':
    xml_path = '/Users/gogleyin/Downloads/gpu-info.xml'
    print process_gpu_xml_file(xml_path)
