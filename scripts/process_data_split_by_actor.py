#! python

# Created by Olivia Roscoe on 27.01.2020

# Imports
import os
import sys
from argparse import ArgumentParser
import shutil

from process_naming import get_actor_from_ravdess_name, get_actor_from_crema_name, \
    is_ravdess_name, is_crema_name, RAVDESS_NUM_ACTORS, CREMA_NUM_ACTORS

if __name__ == "__main__":
    parser = ArgumentParser(description='Split a dataset into train, val, test with 70:20:10 split')
    parser.add_argument('--input_dir', '-i', type=str, help='Folder containing data')
    parser.add_argument('--output_dir', '-o', type=str, default='data_split', help='Folder for output data structure')
    parser.add_argument('--dataset_name', '-d', type=str, default='ravdess', help='Name of dataset: ravdess, crema')
    args = parser.parse_args()

    dir = os.path.abspath(args.input_dir)
    if not os.path.exists(dir):
        print('Provided path', dir, 'is not a valid directory. Please try again')
        sys.exit(-1)

    if args.dataset_name == 'ravdess':
        actor_func = get_actor_from_ravdess_name
        name_func = is_ravdess_name
        actor_num = RAVDESS_NUM_ACTORS
    elif args.dataset_name == 'crema':
        actor_func = get_actor_from_crema_name
        name_func = is_crema_name
        actor_num = CREMA_NUM_ACTORS

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

    # set split params
    train_num = actor_num * 0.6
    val_num = actor_num * 0.2 + train_num

    for filename in os.listdir(dir):  # assuming jpg
        source_path = os.path.join(dir, filename)
        if name_func(filename):
            actor = actor_func(filename)
            if actor <= train_num:
                shutil.move(source_path, train_path)
            elif actor <= val_num:
                shutil.move(source_path, val_path)
            else:
                shutil.move(source_path, test_path)
        else:
            print('Ignoring non-image file ', filename)
