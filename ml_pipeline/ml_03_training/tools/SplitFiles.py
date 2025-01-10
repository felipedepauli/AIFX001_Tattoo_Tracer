import os
import shutil
import random

def create_directory(name):
    if not os.path.exists(name):
        os.mkdir(name)

def distribute_files(files, proportions):
    random.shuffle(files)
    results = []
    start = 0
    for proportion in proportions:
        end = start + int(proportion * len(files))
        results.append(files[start:end])
        start = end
    return results

def main():
    source_dir = '/home/fpauli/git/projects/windows/tattoo_trace/ml_pipeline/datasets/tattoo_trace/detection/tattoo'
    output_dirs = ['train', 'test', 'valid', 'dump']
    proportions = [0.8, 0.2, 0.0]

    for dir_name in output_dirs:
        create_directory(dir_name)

    paired_files = []
    unpaired_files = []
    for filename in os.listdir(source_dir):
        base, ext = os.path.splitext(filename)
        if ext == '.jpg':
            if os.path.exists(os.path.join(source_dir, base + '.json')):
                paired_files.append(filename)
            else:
                unpaired_files.append(filename)

    distributed_files = distribute_files(paired_files, proportions)
    for dir_name, files in zip(output_dirs[:-1], distributed_files):
        for file in files:
            base, ext = os.path.splitext(file)
            shutil.move(os.path.join(source_dir, file), dir_name)
            shutil.move(os.path.join(source_dir, base + '.json'), dir_name)

    for file in unpaired_files:
        shutil.move(os.path.join(source_dir, file), 'dump')

if __name__ == '__main__':
    main()