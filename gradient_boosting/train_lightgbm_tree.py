#! python

# Created by Aleksei Wan on 13.11.2019

# Imports
import os
import sys
import json
from argparse import ArgumentParser
import lightgbm as lgbm
import numpy as np
from random import shuffle

def train_lgbm_model(params, training_data, validation_data=None, save_model_path: str = 'lgbm-model.txt'):
    bst = lgbm.train(params, training_data, valid_sets=[validation_data])
    bst.save_model(save_model_path)
    return bst


def load_lgbm_model(filename: str = 'lgbm-model.txt'):
    return lgbm.Booster(model_file=filename)  # init model


def calculate_acc(predictions, labels):
    total = 0
    acc = 0
    for val in predictions:
        a = np.array(val)
        b = a.argmax()
        if b == labels[total]:
            acc += 1
        total += 1
    acc /= total
    return acc, total


if __name__ == "__main__":
    parser = ArgumentParser(description='Create a LightGBM tree based on provided data')
    parser.add_argument('--input_file', type=str, default='results.json', help='File containing results')
    args = parser.parse_args()
    path_to_check = os.path.abspath(args.input_file)
    if not os.path.exists(path_to_check):
        print('Provided path', path_to_check, 'is not a valid directory. Please try again')
        sys.exit(-1)

    with open(path_to_check, 'r') as file:
        contents = json.load(file)

    shuffle(contents)
    data_list = []
    labels = []
    for element in contents:
        labels.append(element[0])
        data_list.append(element[1])

    idx1 = int(0.8 * len(data_list))
    idx2 = idx1 + int(0.15 * len(data_list))
    train_data = np.array(data_list[:idx1])
    train_labels = np.array(labels[:idx1])
    valid_data = np.array(data_list[idx1:idx2])
    valid_labels = np.array(labels[idx1:idx2])
    test_data = np.array(data_list[idx2:])
    test_labels = np.array(labels[idx2:])

    training_data = lgbm.Dataset(train_data, label=train_labels)
    validation_data = lgbm.Dataset(valid_data, label=valid_labels)

    param = {'objective': 'multiclass',
             'num_class': 8,
             'metric': 'multi_logloss',
             'early_stopping_rounds': 100,
             'max_depth': 8,
             'max_bin': 255,
             'feature_fraction': 1,
             'bagging_fraction': 0.65,
             'bagging_freq': 5,
             'learning_rate': 0.025,
             'num_rounds': 10,
             'num_leaves': 32,
             'min_data_in_leaf': 20,
             'lambda_l1': 0.0001}

    bst = train_lgbm_model(params=param, training_data=training_data, validation_data=validation_data)

    train_acc, train_samples = calculate_acc(bst.predict(train_data), train_labels)
    valid_acc, valid_samples = calculate_acc(bst.predict(valid_data), valid_labels)
    test_acc, test_samples = calculate_acc(bst.predict(test_data), test_labels)

    def make_printable(x): return str(round(100 * float(x), 3))
    print('Training Accuracy', make_printable(train_acc) + '% with', train_samples, 'samples')
    print('Validation Accuracy', make_printable(valid_acc) + '% with', valid_samples, 'samples')
    print('Test Accuracy', make_printable(test_acc) + '% with', test_samples, 'samples')
