import re
from datetime import datetime, timedelta
import numpy as np
import argparse


def _time_delta_in_second(time_str0, time_str1, split='.'):
    format = '%H:%M:%S'
    if split:
        format = '%H:%M:%S' + split + '%f'
    time0 = datetime.strptime(time_str0, format)
    time1 = datetime.strptime(time_str1, format)
    if time0 > time1:
        time0 = time0 - timedelta(hours=24)
    total_seconds = (time1 - time0).total_seconds()
    return total_seconds


def _calculate_average_caffe(lines):
    average = 0.0
    for i in range(1, len(lines)):
        time_delta = _time_delta_in_second(lines[i - 1].split()[1], lines[i].split()[1])
        average += time_delta
    average_loss = 0.0
    for line in lines:
        average_loss += float(line.split()[-1])
    average_loss /= len(lines)
    iterations = int(lines[1].split()[5].strip(',')) - int(lines[0].split()[5].strip(','))
    return average / (len(lines) - 1) / iterations, average_loss


def print_arguments(info):
    return '-t %.6f -a %.6f -I \"%s\"' % info
    # return '-t %.6f -a %.6f -I %s' % info


def extract_info_caffe(filename):
    """
    Use for caffe1.0.0
    """
    f = open(filename)
    content = f.readlines()
    useful_lines = []
    accuracies = []
    is_cpu = False
    average_times = []
    loss_indexes = []
    all_losses = []
    for index, line in enumerate(content):
        if line.find('Use CPU') > 0:
            is_cpu = True
        if line.find('solver.cpp:218') > 0:
            useful_lines.append(line)
            items = line.split(' ')
            iteration_idx = items.index('Iteration') + 1
            iteration = items[iteration_idx]
            average_time = float(items[iteration_idx + 3].split('s/')[0])
            average_times.append(average_time)
            loss = float(items[-1].split('\n')[0])
            all_losses.append(loss)
            # accuracies.append((iteration, '-', loss))
        if line.find('Testing net ') > 0:
            items = line.split(' ')
            iteration_idx = items.index('Iteration') + 1
            loss_indexes.append(int(items[iteration_idx].split(',')[0]))
    start_index = loss_indexes[0]
    for i in loss_indexes[1:]:
        end_index = i
        iteration = i
        loss = np.mean(all_losses[start_index:end_index])
        accuracies.append((iteration, '-', loss))
        start_index = end_index

    # print average_times
    average_time = np.average(average_times)
    try:
        total_time = _time_delta_in_second(content[0].split()[1], content[-1].split()[1])
    except:
        total_time = _time_delta_in_second(content[0].split()[1], content[-3].split()[1])
    seq_str = accuracies
    seq_str = ','.join(['%s:%s:%s' % item for item in accuracies])
    # print seq_str
    return total_time, average_time, seq_str


def extract_info_cntk(filename):
    f = open(filename)
    content = f.readlines()
    useful_lines = []
    for index, line in enumerate(content):
        if line.find('Finished Epoch') >= 0:
            if len(useful_lines) > 0:
                if useful_lines[-1].find(line[0:len('Finished Epoch[23 of 40]')]) < 0:
                    useful_lines.append(line)
            else:
                useful_lines.append(line)

    average_time = 0.0
    accuracies = []
    epoch_size = int(useful_lines[0].split('=')[1].split('*')[1].split(';')[0].strip())
    gpu_count = int(content[-3].split(':')[1])
    batch_size = int(content[-2].split(':')[1])
    actual_batch_size = batch_size * gpu_count
    valid_line = 0
    for index, line in enumerate(useful_lines):
        try:
            time_str = line.split('=')[-1].strip().strip('s')
            average_time += float(time_str)
            valid_line += 1
            loss = float(line.split('=')[1].split('*')[0].strip())
            accuracies.append((index, loss))
        except:
            pass
        # if index % gpu_count == 0:
        #    loss = float(line.split('=')[1].split('*')[0].strip())
        #    accuracies.append((index, loss))

    average_epoch_time = average_time / valid_line
    total_time = float(content[-1].split(':')[1])
    average_batch_time = average_epoch_time / ((epoch_size + actual_batch_size - 1) / actual_batch_size)

    seq_str = ','.join(['%s:-:%s' % item for item in accuracies])
    return total_time, average_batch_time, seq_str


def extract_info_mxnet(filename):
    f = open(filename)
    content = f.readlines()
    average_batch_time = 0.0
    total_time = 0
    seq_str = None
    batch_size = int(content[0].split('=')[1].split(',')[0])
    s = content[0]
    epoch_size = int(s[s.find('num_examples=') + len('num_examples='):].split(',')[0])
    accuracies = []
    for index, line in enumerate(content):
        if line.find('Time cost=') >= 0:
            average_batch_time += float(line.split('=')[1])
        if line.find('Validation-cross-entropy=') >= 0:
            epoch_index = line[line.find('Epoch[') + len('Epoch['):].split(']')[0]
            loss = content[index - 4].split('=')[1].strip('\n')
            accuracy = content[index - 1].split('=')[1].strip('\n')
            accuracies.append((epoch_index, accuracy, loss))

    average_batch_time = (average_batch_time / 40) / (epoch_size / batch_size)
    seq_str = ','.join(['%s:%s:%s' % item for item in accuracies])
    total_time = _time_delta_in_second(content[0].split()[1], content[-1].split()[1], split=',')
    return total_time, average_batch_time, seq_str


def extract_info_tensorflow(filename):
    f = open(filename)
    content = f.readlines()
    average_batch_time = 0
    total_time = 0
    seq_str = None
    for index, line in enumerate(content):
        if line.find('average_batch_time:') >= 0:
            average_batch_time = float(line.split(':')[1])
        if line.find('finished with execute time') >= 0:
            total_time = float(line.split(':')[1].strip())
        if line.find('epoch_info:') >= 0:
            seq_str = line[11:-1].strip()
    if not seq_str:
        try:
            seq_str = '40:%s:-' % content[-1].split('=').strip()
        except:
            seq_str = '0:-:-'
    else:
        seq_split = seq_str.split(',')
        if len(seq_split) > 2000:
            new_seq_split = []
            index = 0
            for i in range(len(seq_split)):
                item = seq_split[i].split(':')
                if int(item[0]) == index:
                    new_seq_split.append(seq_split[i].strip('\''))
                    index += 1
            seq_str = ','.join(new_seq_split)
            # seq_str = new_seq_split
    # return average_batch_time, seq_str
    return average_batch_time


def extract_info_tensorflow_synthetic(filepath):
    # e.g. Forward-backward across 50 steps, 0.698 +/- 0.100 sec / batch
    pattern = 'across \d+ steps, (\d+\.\d+) .+'
    with open(filepath, 'r') as f:
        content = f.read()
        result = re.search(pattern, content)
        average_batch_time = result.group(1)
        return float(average_batch_time)


def extract_info_torch(filename):
    f = open(filename)
    content = f.readlines()
    average_batch_time = 0.0
    total_time = 0
    seq_str = None
    batch_size = 0
    epoch_size = 0
    accuracies = []
    for index, line in enumerate(content):
        if line.find('batchSize') >= 0:
            batch_size = line.split()[-1]
        if line.find('network') >= 0:
            if line.find('alexnet') >= 0 or line.find('resnet') >= 0:
                epoch_size = 50000
            else:
                epoch_size = 60000
        if line.find('Epoch time:') >= 0:
            average_batch_time += float(line.split(':')[-1])
        if line.find('Test accuracy = ') >= 0:
            epoch_index = line.split(':')[3]
            loss = content[index - 15].split(':')[-1].strip('\n').strip()
            accuracy = line.split(':')[-1].strip('\n').strip()
            accuracies.append((epoch_index, accuracy, loss))

    average_batch_time = (average_batch_time / 40) / (int(epoch_size) / int(batch_size))
    # average_batch_time = average_batch_time * int(batch_size) / int(epoch_size)
    seq_str = ','.join(['%s:%s:%s' % item for item in accuracies])
    total_time = _time_delta_in_second(content[0].split()[1], content[-1].split()[1], split=None)
    return total_time, average_batch_time, seq_str


def extract_info(filename, tool, batch_size=32):
    tool = tool.lower()
    if tool == 'caffe':
        return extract_info_caffe(filename)
    elif tool == 'cntk':
        return extract_info_cntk(filename)
    elif tool == 'mxnet':
        return extract_info_mxnet(filename)
    elif tool == 'tensorflow':
        return extract_info_tensorflow(filename)
    elif tool == 'torch':
        return extract_info_torch(filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract information from log file')
    parser.add_argument('-f', '--file', help='Full path of log file')
    parser.add_argument('-t', '--tool', help='Tool name')
    p = parser.parse_args()
    info = extract_info(p.file, p.tool)
    # print info
    print print_arguments(info)
