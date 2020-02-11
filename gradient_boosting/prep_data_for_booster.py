#! python

# Created by Aleksei Wan on 03.02.2020

# Imports
from os import path
from sys import exit, path as sys_path
import json
from argparse import ArgumentParser
from numpy import loadtxt
from bisect import bisect_left
from csv import reader

sys_path.append(path.join(path.dirname(__file__), "../scripts"))
from process_naming import get_emotion_num_from_ravdess_name, aws_number_from_emotion


def read_aws_emotion_confidence(raw_emotions: list) -> list:
    """
    Get AWS Emotion Confidence values
    :param raw_emotions: the raw emotion list from aws
    :return: list of confidence values sorted by emotion number
    """
    l = [(em['Confidence'], aws_number_from_emotion(em['Type'])) for em in raw_emotions]
    l.sort(key=lambda t: t[1])
    return [v[0] for v in l]


def calculate_aws_prediction_list(file_contents) -> list:
    """
    Get AWS Predictions list for merger
    :param file_contents: Loaded AWS JSON
    :return: [filename, emotion confidence values]
    """
    return [(e[0], read_aws_emotion_confidence(e[1]['FaceDetails'][0]['Emotions'])) for e in file_contents]


def calculate_aws_prediction_list_training(file_contents) -> list:
    """
    Get AWS Predictions list for merger
    :param file_contents: Loaded AWS JSON
    :return: [filename, emotion confidence values, label]
    """
    return [
        (e[0], read_aws_emotion_confidence(e[1]['FaceDetails'][0]['Emotions']), get_emotion_num_from_ravdess_name(e[0]))
        for e in file_contents]


def merge_lists(audio_dict: list, img_list: list, training_mode: bool = False) -> list:
    """
    Provide a list of the combined audio & image predictions
    :param audio_dict: {filename: audio emotion confidence values}
    :param img_list: [filename, AWS emotion confidence values, label if training_mode]
    :param training_mode: whether the output should include the label
    :return: [(label, [aws preds, audio preds])] if training_mode else [aws preds, audio preds]
    """
    ret_list = []
    if training_mode:
        for i in img_list:
                if i[0] in audio_dict:
                        ret_list.append((i[2], i[1], audio_dict[i[0]]))

    else:
        for i in img_list:
                if i[0] in audio_dict:
                        ret_list.append((i[1], audio_dict[i[0]]))
    return ret_list
 

if __name__ == "__main__":
    parser = ArgumentParser(description='Create a LightGBM tree based on provided data')
    parser.add_argument('--training', '-t', action='store_true', default=False, help='Flag for training mode')
    parser.add_argument('--audio_input_file', '-a', type=str, default='audio_nn_predictions.csv',
                        help='Audio predictions')
    parser.add_argument('--audio_names_file', '-n', type=str, default='audio_nn_prediction_names.csv',
                        help='Audio names')
    parser.add_argument('--image_input_file', '-i', type=str, default='input.json', help='Image JSON file')
    parser.add_argument('--out_file', '-o', type=str, default='full_results_for_lgbm.json', help="File for output")
    args = parser.parse_args()

    aud_path = path.abspath(args.audio_input_file)
    aud_name_path = path.abspath(args.audio_names_file)
    aws_path = path.abspath(args.image_input_file)
    if not (path.exists(aud_path) and path.exists(aws_path) and path.exists(aud_name_path)):
        print('Provided path was not a valid file or directory. Please try again')
        exit(-1)

    # Read in data
    with open(aws_path, 'r') as file:
        aws_contents = json.load(file)
    aud_preds = loadtxt(aud_path, delimiter=",")
    with open(aud_name_path) as csvfile:
        rows = reader(csvfile, delimiter=",")
        aud_pred_names = [row[0] for row in rows]

    aud_dict = {aud_pred_names[i]: aud_preds[i].tolist() for i in range(len(aud_preds))}
    aws_list = calculate_aws_prediction_list_training(aws_contents) if args.training else calculate_aws_prediction_list(
        aws_contents)

    full_list = merge_lists(aud_dict, aws_list, args.training)
    print('Merged list size:', len(full_list))
    with open(args.out_file, 'w+') as f:
        json.dump(full_list, f)
