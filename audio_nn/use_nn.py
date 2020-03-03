#! python

# Created by Olivia Roscoe on 28.01.2020

from os import path
from sys import exit
import audio_nn as nn
from argparse import ArgumentParser
from numpy import savetxt

if __name__ == "__main__":
    parser = ArgumentParser(description='Get predictions from audio neural net')
    parser.add_argument('--audio_dir', '-a', type=str, help='Dir containing audio spectograms for classification',
                        required=True)
    parser.add_argument('--checkpoint', '-c', type=str, help="File path for checkpoint to load model from")
    parser.add_argument('--batch_size', '-b', type=int, default=500, help="Number of images per batch")
    args = parser.parse_args()

    # Check that paths are valid
    if args.checkpoint is None:
        print('Must provide a checkpoint file to run an evaluation. Please try again')
        exit(-1)
    if not path.isfile(args.checkpoint):
        print('Provided path', args.checkpoint, 'is not a valid file. Please try again')
        exit(-1)
    if not path.exists(args.audio_dir):
        print('Provided path', args.audio_dir, 'is not a valid directory. Please try again')
        exit(-1)

    model = nn.AudioNN()
    model.load_from_checkpoint(args.checkpoint)

    dataloader = nn.load_data_with_filename(args.audio_dir, args.batch_size, model)
    pred = model.get_prediction(dataloader)[0]

    # do something with pred, for now just save to CSV
    savetxt('audio_nn_predictions.csv', pred[0].numpy(), delimiter=",", fmt='%s')
    savetxt('audio_nn_prediction_names.csv', pred[1], delimiter=",", fmt='%s')