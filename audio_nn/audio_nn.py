#! python

# Created by Olivia Roscoe on 25.11.2019

import os
import re
from PIL import Image
from sys import path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as utils
from torchvision import transforms, models

# TODO: Look for better way to deal with hardcoded filepath
path.insert(1, os.path.join(path[0], '../scripts'))
import process_naming as pn

from datetime import datetime
import numpy as np

# regex for image file matching
IMG_FILE = re.compile('(.*)rgb_plt(.*).png$')


# 2 layer linear fully connected neural network
class AudioClassifier(nn.Module):

    # assume input image is of size 4 x 200 x 140
    def __init__(self):
        super(AudioClassifier, self).__init__()
        self.name = "audio"

        self.fc1 = nn.Linear(256 * 5 * 3, 1050)
        self.fc2 = nn.Linear(1050, 380)
        self.fc3 = nn.Linear(380, 8)

        self.drop = nn.Dropout(p=0.15)

    # inputs to the nn must be tensors of form [N, C, H, W]
    def forward(self, x):
        x = x.view(-1, 256 * 5 * 3)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(self.drop(x)))
        x = self.fc3(self.drop(x))
        return x


def generate_data_label(filename: str) -> torch.tensor:
    if pn.is_ravdess_name(filename):
        emotion = pn.get_emotion_num_from_ravdess_name(filename)
    elif pn.is_crema_name(filename):
        emotion = pn.get_emotion_num_from_crema_name(filename)
    return torch.tensor(emotion)


def load_data(directory, batch_size):
    image_list = []
    labels = []
    # padding is the colour of silence in the clips
    data_transform = transforms.Compose([transforms.Pad((60, 0), fill=(0, 127, 127)),
                                         transforms.CenterCrop((200, 140)),
                                         transforms.ToTensor()])

    print('Using', directory, 'as audio directory... ')

    for filename in os.listdir(directory):  # assuming jpg
        if re.match(IMG_FILE, filename) is not None:
            # Transforms: i) Add padding to horizontal dimension of image
            #            ii) crop to be 140x200
            #           iii) convert image to tensor
            image_list.append(data_transform(Image.open(os.path.join(directory, filename)).convert('RGB')))
            labels.append(generate_data_label(filename))
        else:
            print('Ignoring non-image file', filename)

    tensor_img = torch.stack(image_list)
    tensor_labels = torch.stack(labels)

    my_dataset = utils.TensorDataset(tensor_img, tensor_labels)  # create dataset
    my_dataloader = utils.DataLoader(my_dataset, batch_size=batch_size)

    return my_dataloader


def load_data_with_filename(directory, batch_size, network):
    image_list = []
    image_name = []
    # padding is the colour of silence in the clips
    data_transform = transforms.Compose([transforms.Pad((60, 0), fill=(0, 127, 127)),
                                         transforms.CenterCrop((200, 140)),
                                         transforms.ToTensor()])

    print('Using', directory, 'as audio directory... ')

    for filename in os.listdir(directory):  # assuming jpg
        match_obj = re.match(IMG_FILE, filename)
        if match_obj is not None:
            image_list.append(data_transform(Image.open(os.path.join(directory, filename))))
            image_name.append(match_obj[1])
        else:
            print('Ignoring non-image file', filename)

    tensor_img = torch.stack(image_list)
    tensor_names = torch.from_numpy(np.linspace(0, len(image_name), len(image_name)))
    network.name_list = image_name

    my_dataset = utils.TensorDataset(tensor_img, tensor_names)  # create dataset
    my_dataloader = utils.DataLoader(my_dataset, batch_size=batch_size)

    return my_dataloader


class AudioNN:
    def __init__(self, name="model"):
        file_path = os.path.dirname(os.path.realpath(__file__))
        self.model_name = name
        self.checkpoint_dir = os.path.join(file_path, 'checkpoints')
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        self.model = AudioClassifier()
        self.name_list = []
        # self.use_cuda = torch.cuda.is_available()
        # if self.use_cuda:
        #     print("Using cuda")
        #     self.model.cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        self.alexnet = models.alexnet(pretrained=True)

    def load_data(self, directory, batch_size):
        image_list = []
        labels = []
        # padding is the colour of silence in the clips
        data_transform = transforms.Compose([transforms.Pad((60, 0), fill=(0, 127, 127)),
                                             transforms.CenterCrop((200, 140)),
                                             transforms.ToTensor()])

        print('Using', directory, 'as audio directory... ')

        for filename in os.listdir(directory):  # assuming jpg
            img_file = re.match(IMG_FILE, filename)
            if img_file:
                img_path = os.path.join(directory, filename)
                im = Image.open(img_path).convert('RGB')
                # Transforms: i) Add padding to horizontal dimension of image
                #            ii) crop to be 140x200
                #           iii) convert image to tensor
                im = data_transform(im)
                image_list.append(im)

                # TODO: will need to generalize once more datasets are in use
                emotion = get_emotion_num_from_ravdess_name(filename)
                labels.append(torch.tensor(emotion))
            else:
                print('Ignoring non-image file ', filename)

        tensor_img = torch.stack(image_list)
        tensor_labels = torch.stack(labels)

        my_dataset = utils.TensorDataset(tensor_img, tensor_labels)  # create your datset
        my_dataloader = utils.DataLoader(my_dataset, batch_size=batch_size, shuffle=True)

        return my_dataloader

    def train_network(self, train, batch_size, num_epochs, valid):
        criterion = nn.CrossEntropyLoss()
        train_acc, val_acc = np.zeros(num_epochs), np.zeros(num_epochs)
        checkpoint_datetime = datetime.now().strftime("%Y%m%d-%H%M%S")

        # training
        for epoch in range(num_epochs):
            print("Training epoch", epoch)
            for imgs, labels in train:
                imgs = self.alexnet.features(imgs)
                # if self.use_cuda:
                #     imgs = imgs.cuda()
                #     labels = labels.cuda()

                self.model.train()
                out = self.model(imgs)  # forward pass
                loss = criterion(out, labels)  # compute the total loss
                loss.backward()  # backward pass (compute parameter updates)
                self.optimizer.step()  # make the updates for each parameter
                self.optimizer.zero_grad()  # a clean up step for PyTorch

            # save the current training information every iteration
            #losses.append(float(loss) / batch_size)  # compute *average* loss
            train_acc[epoch] = self.get_accuracy(train)  # compute training accuracy
            if valid is not None:
                val_acc[epoch] = self.get_accuracy(valid)  # compute validation accuracy

            # save the current model parameters
            device = "cpu"
            # if self.use_cuda:
            #     device = "gpu"

            model_info = checkpoint_datetime + "_{0}_bs{1}_epoch{2}_{3}".format(self.model_name, batch_size, epoch, device)
            checkpoint_path = os.path.join(self.checkpoint_dir, model_info)
            self.checkpoint_model(epoch, loss, checkpoint_path)

            np.savetxt(os.path.join(self.checkpoint_dir, checkpoint_datetime + "_acc.csv"), train_acc, delimiter=",", fmt='%s')
            #np.savetxt(os.path.join(self.checkpoint_dir, checkpoint_datetime + "_loss.csv"), losses, delimiter=",",
            #           fmt='%s')

            if valid is not None:
                np.savetxt(os.path.join(self.checkpoint_dir, checkpoint_datetime + "_val_acc.csv"), val_acc,
                           delimiter=",", fmt='%s')

        print("Final Training Accuracy: {}".format(train_acc[-1]))

    def get_accuracy(self, data):
        correct = 0
        total = 0
        self.model.eval()
        for imgs, labels in data:
            imgs = self.alexnet.features(imgs)
            # if self.use_cuda:
            #     imgs = imgs.cuda()
            #     labels = labels.cuda()
            output = self.model(imgs)

            # select index with maximum prediction score
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(labels.view_as(pred)).sum().item()
            total += imgs.shape[0]
        return correct / total

    def checkpoint_model(self, epoch, loss, path):
        # save the current model state to a file
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }, path)

    def load_from_checkpoint(self, path):
        # load the model state from a final
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        self.model.eval()

        return epoch, loss

    def get_prediction(self, data):
        self.model.eval()
        predictions = []
        for imgs, name_ids in data:
            imgs = self.alexnet.features(imgs)
            # if self.use_cuda:
                # imgs = imgs.cuda()

            output = self.model(imgs)
            predictions.append((output.data, [self.name_list[i] for i in range(len(imgs))]))
        return predictions
