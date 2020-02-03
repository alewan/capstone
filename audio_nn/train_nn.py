#! python

# Created by Olivia Roscoe on 28.01.2020

from os import path
from sys import exit
import audio_nn as nn
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser(description='Train the audio neural net')
    parser.add_argument('--audio_dir', '-a', type=str, help='Dir containing audio spectograms for classification',
                        required=True)
    parser.add_argument('--val_dir', '-v', type=str, help='Dir containing validation data')
    parser.add_argument('--model_name', '-m', type=str, help='Name of the model for checkpointing', required=True)
    parser.add_argument('--batch_size', '-b', type=int, default=500, help="Number of images per batch")
    parser.add_argument('--epochs', '-e', type=int, default=100, help="Number of training epochs")
    args = parser.parse_args()

    if not path.exists(args.audio_dir):
        print('Provided path', args.audio_dir, 'is not a valid directory. Please try again')
        exit(-1)

    model = nn.AudioNN(args.model_name)

    dataloader = nn.load_data(args.audio_dir, args.batch_size)

    # set up the validation data loader if its being used
    val_loader = None
    if args.val_dir is not None:
        if not path.exists(args.val_dir):
            print('Provided validation path', args.val_dir, 'is not a valid directory. Please try again')
            exit(-1)
        val_loader = nn.load_data(args.val_dir, args.batch_size)

    model.train_network(dataloader, batch_size=args.batch_size, num_epochs=args.epochs, valid=val_loader)
