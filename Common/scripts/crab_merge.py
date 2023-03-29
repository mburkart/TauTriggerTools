#!/usr/bin/env python
# Submit jobs on CRAB.
# This file is part of https://github.com/cms-tau-pog/TauTriggerTools.

import argparse
import os
import re
import subprocess
import sys

parser = argparse.ArgumentParser(description='Merge outputs of the finished CRAB jobs.')
parser.add_argument('crabOutput', nargs=1, type=str, help="Path with output of crab jobs")
parser.add_argument('mergedOutput', nargs=1, type=str, help="Path where to store merged outputs")
args = parser.parse_args()

input = args.crabOutput[0]
output = args.mergedOutput[0]
if not os.path.exists(output):
    os.makedirs(output)
output = os.path.abspath(output)

merged_jobs = []
skipped_jobs = []
for dataset_dir in sorted(os.listdir(input)):
    dataset_path = os.path.join(input, dataset_dir)
    if not os.path.isdir(dataset_path): continue
    for crab_name_dir in sorted(os.listdir(dataset_path)):
        crab_name_path = os.path.join(dataset_path, crab_name_dir)
        if not os.path.isdir(crab_name_path): continue
        crab_name = re.sub(r'^crab_', '', crab_name_dir)
        output_name = crab_name + '.root'
        output_path = os.path.join(output, output_name)
        if os.path.isfile(output_path):
            print('{} already exists.'.format(output_name))
            skipped_jobs.append(crab_name)
            continue
        print('Merging "{}" ...'.format(output_name))
        crab_job_ids = os.listdir(crab_name_path)
        if len(crab_job_ids) != 1:
            raise RuntimeError('More than 1 job id for crab job {} is present.'.format(crab_name_path))
        crab_job_id_path = os.path.join(crab_name_path, crab_job_ids[0])
        root_files = []
        for file_block in os.listdir(crab_job_id_path):
            file_block_path = os.path.join(crab_job_id_path, file_block)
            if not os.path.isdir(file_block_path): continue
            for file in os.listdir(file_block_path):
                file_full_path = os.path.join(file_block_path, file)
                if not os.path.isfile(file_full_path) or not file.endswith('.root'): continue
                root_files.append(os.path.join(file_block, file))

        cmd = 'cd {} ; hadd -O -ff -n 11 {} {}'.format(crab_job_id_path, output_path, ' '.join(root_files))
        result = subprocess.call([cmd], shell=True)
        if result != 0:
            raise RuntimeError('Failed to merge "{}" into "{}"'.format(crab_job_id_path, output_path))
        merged_jobs.append(crab_name)
if len(skipped_jobs):
    print('Following jobs were skipped, because corresponding file already exists: {}'.format(' '.join(skipped_jobs)))
if len(merged_jobs):
    print('Following jobs has been merged: {}'.format(' '.join(merged_jobs)))
