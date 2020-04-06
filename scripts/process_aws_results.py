#! python

# Created by Aleksei Wan on 13.11.2019

# Imports
import os
import sys
from argparse import ArgumentParser
import json

sys.path.append('..')
from scripts.process_naming import get_emotion_from_ravdess_name_aws


# Takes in raw AWS emotions data struct, returns list sorted by confidence (highest to lowest)
def read_aws_emotions(raw_emotions: list) -> list:
    ret_list = [(em['Type'], em['Confidence']) for em in raw_emotions]
    ret_list.sort(key=lambda t: t[1], reverse=True)
    return ret_list


def first_element_list_from_tuple_list(in_list: list) -> list:
    return [t[0] for t in in_list]


if __name__ == "__main__":
    parser = ArgumentParser(description='Process results from AWS to determine correct classification')
    parser.add_argument('--input_file', type=str, default='results.json', help='File containing results')
    args = parser.parse_args()

    path_to_check = os.path.abspath(args.input_file)
    if not os.path.exists(path_to_check):
        print('Provided path', path_to_check, 'is not a valid directory. Please try again')
        sys.exit(-1)

    with open(path_to_check, 'r') as file:
        contents = json.load(file)

    total = 0
    direct_match_counter = 0
    second_match_counter = 0
    emotion_not_covered = 0
    for e in contents:
        actual_emotion = get_emotion_from_ravdess_name_aws(e[0])
        aws_emotions = first_element_list_from_tuple_list(read_aws_emotions(e[1]['FaceDetails'][0]['Emotions']))
        # Check if top emotion matches
        print('Actual:', actual_emotion, '\nAWS:', aws_emotions)
        if actual_emotion == aws_emotions[0]:
            direct_match_counter += 1
        elif actual_emotion == aws_emotions[1]:
            second_match_counter += 1
        else:
            if actual_emotion not in aws_emotions:
                emotion_not_covered += 1
        total += 1

    total_for_comp = float(total - emotion_not_covered)
    print('Number of samples:', total_for_comp)
    def make_printable(x): return round(100 * float(x), 3)
    print('Number without direct match:', emotion_not_covered)
    print('Total Accuracy', make_printable(direct_match_counter / total_for_comp), '%')
    print('Soft Match Accuracy', make_printable((direct_match_counter + second_match_counter) / total_for_comp), '%')
