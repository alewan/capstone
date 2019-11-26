#! python

# Created by Aleksei Wan on 26.11.2019

# Imports
import os
import sys
from argparse import ArgumentParser
import json

from process_ravdess_naming import get_emotion_num_from_ravdess_name, aws_number_from_emotion
from process_aws_results import first_element_list_from_tuple_list


# Takes in raw AWS emotions data struct, returns list sorted by confidence (highest to lowest)
def read_aws_emotions(raw_emotions: list) -> list:
    ret_list = []
    for e in raw_emotions:
        to_append = (e['Confidence'], aws_number_from_emotion(e['Type']))
        if to_append[1] != -1:
            ret_list.append(to_append)
    ret_list.sort(key=lambda t: t[1])
    return ret_list


if __name__ == "__main__":
    parser = ArgumentParser(description='Process results from AWS to prep for feeding into LightGBM')
    parser.add_argument('--input_file', type=str, default='results.json', help='File containing results')
    args = parser.parse_args()

    path_to_check = os.path.abspath(args.input_file)
    if not os.path.exists(path_to_check):
        print('Provided path', path_to_check, 'is not a valid directory. Please try again')
        sys.exit(-1)

    with open(path_to_check, 'r') as file:
        contents = json.load(file)

    dump_list = []
    for e in contents:
        dump_list.append((get_emotion_num_from_ravdess_name(e[0]),
                          first_element_list_from_tuple_list(read_aws_emotions(e[1]['FaceDetails'][0]['Emotions']))))

    with open('full_results_for_lgbm.json', 'w+') as f:
        json.dump(dump_list, f)
