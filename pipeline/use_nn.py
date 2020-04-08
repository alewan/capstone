#! python

# Created by Olivia Roscoe on 28.01.2020

from os import path
from sys import exit
import audio_nn as nn
from argparse import ArgumentParser
import numpy as np
from scipy.special import softmax

from numpy import savetxt
import json


def audio_neural_network(checkpoint, audio_dir, batch_size):
    # Check that paths are valid
    if checkpoint is None:
        print('Must provide a checkpoint file to run an evaluation. Please try again')
        exit(-1)
    if not path.isfile(checkpoint):
        print('Provided path', checkpoint, 'is not a valid file. Please try again')
        exit(-1)
    if not path.exists(audio_dir):
        print('Provided path', audio_dir, 'is not a valid directory. Please try again')
        exit(-1)

    model = nn.AudioNN()
    model.load_from_checkpoint(checkpoint)

    dataloader = nn.load_data_with_filename(audio_dir, batch_size, model)
    pred = model.get_prediction(dataloader)[0]

    # do something with pred, for now just save to CSV
    savetxt('../pipeline/pipeline_helper/audio_nn_predictions.csv', pred[0].numpy(), delimiter=",", fmt='%s')
    savetxt('../pipeline/pipeline_helper/audio_nn_prediction_names.csv', pred[1], delimiter=",", fmt='%s')

    # softmax nn predictions in order to display on front end 
    nn_prd = pred[0].numpy() 

    # # new array for softmaxed prediction values
    sm_pred = np.zeros((4, 8))
    
    # sm = softmax(nn_prd)

    i = 0
    for row in nn_prd:
        sm_pred[i] = np.exp(row - np.max(row))
        sm_pred[i] = sm_pred[i] / sm_pred[i].sum()
        i += 1

    # find mean of predictions accross all audio predictions
    sm_pred_mean = np.zeros(8)
    sm_pred_mean = np.mean(sm_pred, axis=0) * 100

    # put the audio nn predictions into a json for pipeline display on front end
    pred_list = sm_pred_mean.tolist()    
    with open('../pipeline/pipeline_helper/audio_nn_predictions.json', 'w+') as f:
        json.dump(pred_list, f)

if __name__ == "__main__":
    parser = ArgumentParser(description='Get predictions from audio neural net')
    parser.add_argument('--audio_dir', '-a', type=str, help='Dir containing audio spectograms for classification',
                        required=True)
    parser.add_argument('--checkpoint', '-c', type=str, help="File path for checkpoint to load model from")
    parser.add_argument('--batch_size', '-b', type=int, default=500, help="Number of images per batch")
    args = parser.parse_args()

    audio_neural_network(args.checkpoint, args.audio_dir, args.batch_size)

    

   