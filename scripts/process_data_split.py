#! python

# Created by Olivia Roscoe on 27.01.2020

# Imports
import os
import random
import sys
from argparse import ArgumentParser
import shutil

from process_naming import is_ravdess_name, is_crema_name

if __name__ == "__main__":
    parser = ArgumentParser(description='Split a dataset into train, val, test with 70:20:10 split')
    parser.add_argument('--input_dir', '-i', type=str, help='Folder containing data', required=True)
    parser.add_argument('--output_dir', '-o', type=str, default='data_split', help='Folder for output data structure')
    args = parser.parse_args()

    dir = os.path.abspath(args.input_dir)
    if not os.path.exists(dir):
        print('Provided path', dir, 'is not a valid directory. Please try again')
        sys.exit(-1)

    train_path = os.path.join(args.output_dir, 'train')
    val_path = os.path.join(args.output_dir, 'val')
    test_path = os.path.join(args.output_dir, 'test')

    # Create the new directory structure
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
        os.mkdir(train_path)
        os.mkdir(val_path)
        os.mkdir(test_path)
    else:
        if not os.path.exists(train_path):
            os.mkdir(train_path)
        if not os.path.exists(val_path):
            os.mkdir(val_path)
        if not os.path.exists(test_path):
            os.mkdir(test_path)

    files = os.listdir(dir)
    num_files = len(files)

    # this is to get a mix of actors and emotions
    random.shuffle(files)

    # set split params
    train_num = num_files * 0.6
    val_num = num_files * 0.2 + train_num

    i = 0
    for filename in files:  # assuming jpg
        source_path = os.path.join(dir, filename)
        if is_ravdess_name(filename) or is_crema_name(filename):
            if i <= train_num:
                shutil.move(source_path, train_path)
            elif i <= val_num:
                shutil.move(source_path, val_path)
            else:
                shutil.move(source_path, test_path)
        else:
            print('Ignoring non-image file ', filename)
        i += 1
