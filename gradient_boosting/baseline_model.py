import os
import sys
import json
from argparse import ArgumentParser
import numpy as np


def make_printable(x: float) -> str:
    return str(round(100.0 * x, 3)) + '%'


if __name__ == "__main__":
    parser = ArgumentParser(description='Evaluate a baseline model based on provided data')
    parser.add_argument('--input_file', '-i', type=str, default='results.json', help='File containing results')
    args = parser.parse_args()
    path_to_check = os.path.abspath(args.input_file)
    if not os.path.exists(path_to_check):
        print('Provided path', path_to_check, 'is not a valid directory. Please try again')
        sys.exit(-1)

    # Read Input JSON file
    with open(path_to_check, 'r') as file:
        contents = json.load(file)

    correct = 0
    count = 0
    for elem in contents:
        label = elem[0]
        data = np.zeros(8)
        for idx in range(8):
            data[idx] = (elem[1][idx] + elem[1][idx + 8]) * 0.5
        if np.argmax(data) == label:
            correct += 1
        count += 1
    print('Percentage Accuracy of Baseline', make_printable(correct / count))
