from sys import argv
from glob import glob
import os
import shutil

folders = argv[1:]

split_ways = 4
output_dir = 'split'

for folder in folders:
    folder_files = sorted(glob(folder + '/*.jpg'))
    step = len(folder_files) // 4

    for i in range(split_ways):
        target_dir = os.path.join(output_dir, 'split_{}'.format(i + 1), os.path.split(folder)[-1])
        os.makedirs(target_dir, exist_ok=True)
        for file in folder_files[i * step: (i + 1) * step]:
            shutil.copy(file, target_dir)
