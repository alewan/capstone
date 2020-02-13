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

sys.path.append(os.path.join(os.path.dirname(__file__), "../scripts"))
from perturb_parameters import generate_bdst_param_pack


def train_lgbm_model(params, training_data, validation_data=None, save_model_path: str = 'lgbm-model.txt'):
    bst = lgbm.train(params, training_data, valid_sets=[validation_data])
    bst.save_model(save_model_path)
    return bst


def calculate_acc(predictions, labels):
    acc = 0
    for idx, val in enumerate(predictions):
        if np.argmax(val) == labels[idx]:
            acc += 1
    acc /= len(labels)
    return acc


def make_printable(x) -> str: return str(round(100 * float(x), 3))


if __name__ == "__main__":
    parser = ArgumentParser(description='Create a LightGBM tree based on provided data')
    parser.add_argument('--input_file', '-i', type=str, default='results.json', help='File containing results')
    parser.add_argument('--epochs', '-e', type=int, default=3, help='File containing results')
    args = parser.parse_args()
    path_to_check = os.path.abspath(args.input_file)
    if not os.path.exists(path_to_check):
        print('Provided path', path_to_check, 'is not a valid directory. Please try again')
        sys.exit(-1)

    # Read Input JSON file
    with open(path_to_check, 'r') as file:
        contents = json.load(file)

    # Create data sets
    shuffle(contents)
    data_list = []
    labels = []
    for element in contents:
        labels.append(element[0])
        data_list.append(element[1])
    idx1 = int(0.8 * len(data_list))
    idx2 = idx1 + int(0.1 * len(data_list))
    train_data = np.array(data_list[:idx1])
    train_labels = np.array(labels[:idx1])
    valid_data = np.array(data_list[idx1:idx2])
    valid_labels = np.array(labels[idx1:idx2])
    test_data = np.array(data_list[idx2:])
    test_labels = np.array(labels[idx2:])
    training_data = lgbm.Dataset(train_data, label=train_labels)
    validation_data = lgbm.Dataset(valid_data, label=valid_labels)

    print('Training Samples:', len(train_labels))
    print('Validation Samples:', len(train_labels))
    print('Testing Samples:', len(train_labels))

    # Run training
    acc_list = np.zeros((args.epochs, 3))
    param_list = []
    for i in range(args.epochs):
        param_list.append(generate_bdst_param_pack())

        bst = train_lgbm_model(params=param_list[i], training_data=training_data, validation_data=validation_data,
                               save_model_path='lgbm-model_' + str(i) + '.txt')

        train_acc = calculate_acc(bst.predict(train_data), train_labels)
        valid_acc = calculate_acc(bst.predict(valid_data), valid_labels)
        test_acc = calculate_acc(bst.predict(test_data), test_labels)

        acc_list[i][0] = calculate_acc(bst.predict(train_data), train_labels)
        acc_list[i][1] = calculate_acc(bst.predict(valid_data), valid_labels)
        acc_list[i][2] = calculate_acc(bst.predict(test_data), test_labels)

    np.savetxt('lgbm_model_accuracies.csv', acc_list, delimiter=",", fmt='%s')
    with open('lgbm_model_params.txt', 'w') as outfile:
        json.dump(param_list, outfile)

    vt_acc_list = np.array([0.5 * (a[1] + a[2]) for a in acc_list.tolist()])

    best_tree_idx = np.argmax(vt_acc_list)
    print('Best Tree:', best_tree_idx, '(with validation/test accuracy',
          make_printable(vt_acc_list[best_tree_idx]) + '%)')
    print('Best Tree Params:', param_list[best_tree_idx], '\n')

    vt_acc_list[best_tree_idx] = 0
    best_tree_idx = np.argmax(vt_acc_list)
    print('Second Best Tree:', best_tree_idx, '(with validation/test accuracy',
          make_printable(vt_acc_list[best_tree_idx]) + '%)')
    print('Second Best Tree Params:', param_list[best_tree_idx])

    vt_acc_list[best_tree_idx] = 0
    best_tree_idx = np.argmax(vt_acc_list)
    print('Third Best Tree:', best_tree_idx, '(with validation/test accuracy',
          make_printable(vt_acc_list[best_tree_idx]) + '%)')
    print('Third Best Tree Params:', param_list[best_tree_idx])
