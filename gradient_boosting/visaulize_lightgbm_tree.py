#! python

# Created by Aleksei Wan on 04.02.2020

# Imports
import os
from sys import exit
from argparse import ArgumentParser
import lightgbm as lgbm

if __name__ == "__main__":
    parser = ArgumentParser(description='Create a LightGBM tree based on provided data')
    parser.add_argument('--model', '-m', type=str, default='lgbm-model.txt', help='File containing pre-trained model')
    args = parser.parse_args()

    path_to_model = os.path.abspath(args.model)
    if not os.path.exists(path_to_model):
        print('Provided model path was not a valid file or directory. Please try again')
        exit(-1)

    # Instantiate LightGBM model from saved file
    bst = lgbm.Booster(model_file=path_to_model)

    # Saves diagraph to file
    tree_name = os.path.splitext(os.path.split(path_to_model)[1])[0]
    (lgbm.create_tree_digraph(bst)).render(cleanup=True, filename='diagraph_' + tree_name)
    exit(0)
