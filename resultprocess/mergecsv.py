#!/usr/bin/env python
# coding=utf-8

""" mergecsv.py: Handle result csv file.

Created by gogleyin on 4/8/18.
"""

import os
import csv
import logging

def merge_csvs(files, identifiers, new_field_name, output_file):
    headline = None
    final_rows = []
    for index, f in enumerate(files):
        if not f or not os.path.isfile(f):
            logging.warning('File not exist: %s' % f)
            continue
        identifier = identifiers[index]
        with open(f, 'rb') as csv_file:
            first_line = csv_file.readline()
            if not headline:
                headline = '%s,%s' % (new_field_name, first_line)
                final_rows.append(headline)
            rows = csv_file.readlines()
            rows = ['%s,%s' % (identifier, row) for row in rows]
            final_rows += rows
    with open(output_file, 'wb') as output:
        output.writelines(final_rows)

if __name__ == '__main__':
    csvs = [
        '/Users/gogleyin/Documents/work/cloud/testdata/al-m40/GpuBenchmarkLog_180330-194133/all_results.csv',
        '/Users/gogleyin/Documents/work/cloud/testdata/al-p100/GpuBenchmarkLog_180330-200111/all_results.csv',
        '/Users/gogleyin/Documents/work/cloud/testdata/tx-m40/GpuBenchmarkLog_180330-164749/all_results.csv',
        '/Users/gogleyin/Documents/work/cloud/testdata/tx-p40/GpuBenchmarkLog_180330-194439/all_results.csv',
    ]
    output_csv = '/Users/gogleyin/Desktop/all.csv'
    ids = ['al-m40', 'al-p100', 'tx-m40', 'tx-p40']
    merge_csvs(csvs, ids, 'gpu', output_csv)



