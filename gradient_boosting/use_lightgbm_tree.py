#! python

# Created by Aleksei Wan on 27.01.2020

# Imports (only necessary functions)
from os import path
from sys import exit
from json import load as load_from_json_file
from argparse import ArgumentParser
import lightgbm as lgbm
from numpy import array, argmax

if __name__ == "__main__":
    parser = ArgumentParser(description='Create a LightGBM tree based on provided data')
    parser.add_argument('--model', '-m', type=str, default='lgbm-model.txt', help='File containing pre-trained model')
    parser.add_argument('--input_file', '-i', type=str, default='input.json', help='File containing inputs')
    args = parser.parse_args()
    path_to_check = path.abspath(args.input_file)
    if not (path.exists(path_to_check) and path.exists(path.abspath(args.model))):
        print('Provided path was not a valid file or directory. Please try again')
        exit(-1)

    # Read in data
    with open(path_to_check, 'r') as file:
        data_list = load_from_json_file(file)
    data_list = array(data_list)

    # Instantiate LGBM model from saved file
    bst = lgbm.Booster(model_file=args.model)

    # Make predictions
    raw_preds = bst.predict(data_list)
    final_preds = argmax(raw_preds, 1)

    # Output predictions
    print(final_preds)
