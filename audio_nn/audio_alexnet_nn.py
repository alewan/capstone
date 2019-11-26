#! python

import os
import re
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from PIL import Image
from sys import exit

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as utils
from torchvision import models, transforms

# regex for image file matching
IMG_FILE = re.compile('(.*)\.jp[e]?g$')


# 2 layer linear fully connected neural network
class AudioClassifierAlexNet(nn.Module):

    def __init__(self):
        super(AudioClassifierAlexNet, self).__init__()
        # 256 * 11 * 6 is the output size of running the 3 x 396 x 224
        # image through alexnet feature extraction
        self.fc1 = nn.Linear(256 * 11 * 6, 50)
        self.fc2 = nn.Linear(50, 8)

    # inputs to the nn must be tensors of form [N, C, H, W]
    def forward(self, x):
        x = x.view(-1, 256 * 11 * 6)  # flatten feature data
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class AudioNN:

    def __init__(self, name):
        file_path = os.path.dirname(os.path.realpath(__file__))
        self.model_name = name
        self.checkpoint_dir = os.path.join(file_path, 'checkpoints')
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        self.alexnet = models.alexnet(pretrained=True)
        self.classifier = AudioClassifierAlexNet()
        self.optimizer = optim.SGD(self.classifier.parameters(), lr=0.01, momentum=0.9)

    def load_data(self, directory, batch_size=80):
        image_list = []
        labels = []
        data_transform = transforms.Compose([transforms.Resize(224),
                                             transforms.ToTensor()])

        print('Using ' + directory + ' as audio directory... ')

        for filename in os.listdir(directory):  # assuming jpg
            img_file = re.match(IMG_FILE, filename)
            if img_file:
                img_path = os.path.join(directory, filename)
                im = Image.open(img_path)
                # Transforms: i) convert image to tensor
                #            ii) resize smaller edge to 224
                im = data_transform(im)
                image_list.append(im)
                cats = filename.split("-")
                labels.append(torch.tensor(int(cats[2]) - 1))
            else:
                print('Ignoring non-image file ' + filename)

        tensor_img = torch.stack(image_list)
        tensor_labels = torch.stack(labels)

        my_dataset = utils.TensorDataset(tensor_img, tensor_labels)  # create your datset
        my_dataloader = utils.DataLoader(my_dataset, batch_size=batch_size)

        return my_dataloader

    def get_features(self, dataset):
        features = []
        for i, data in enumerate(dataset, 0):
            inputs, labels = data
            f = self.alexnet.features(inputs)
            f = torch.from_numpy(f.detach().numpy())
            features.append((f, labels))
        return features

    def train_with_features(self, train_f, valid_f=None, batch_size=500, num_epochs=100):

        criterion = nn.CrossEntropyLoss()
        iters, losses, train_acc, val_acc = [], [], [], []

        # training
        n = 0  # the number of iterations
        for epoch in range(num_epochs):
            for features, labels in train_f:

                self.classifier.train()
                out = self.classifier(features)  # forward pass
                loss = criterion(out, labels)  # compute the total loss
                loss.backward()  # backward pass (compute parameter updates)
                self.optimizer.step()  # make the updates for each parameter
                self.optimizer.zero_grad()  # a clean up step for PyTorch

                # save the current training information every iteration
                iters.append(n)
                losses.append(float(loss) / batch_size)  # compute *average* loss
                train_acc.append(self.get_accuracy(train_f))  # compute training accuracy
                if valid_f is not None:
                    val_acc.append(self.get_accuracy(valid_f))  # compute validation accuracy
                n += 1

            # save the current model parameters
            model_info = "model_{0}_bs{1}_epoch{2}".format(self.model_name, batch_size, epoch)
            checkpoint_path = os.path.join(self.checkpoint_dir, model_info)
            self.checkpoint_model(epoch, loss, checkpoint_path)

        # plotting
        plt.title("Training Curve")
        plt.plot(iters, losses, label="Train")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.show()

        plt.title("Training Curve")
        plt.plot(iters, train_acc, label="Train")
        plt.xlabel("Iterations")
        plt.ylabel("Training Accuracy")
        plt.legend(loc='best')
        plt.show()

        print("Final Training Accuracy: {}".format(train_acc[-1]))

    def get_accuracy(self, data):

        correct = 0
        total = 0
        self.classifier.eval()
        for features, labels in data:
            output = self.classifier(features)

            # select index with maximum prediction score
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(labels.view_as(pred)).sum().item()
            total += features.shape[0]
        return correct / total

    def checkpoint_model(self, epoch, loss, path):
        # save the current model state to a file
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.classifier.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }, path)

    def load_from_checkpoint(self, path):
        # load the model state from a final
        checkpoint = torch.load(path)
        self.classifier.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        self.classifier.load_state_dict(torch.load(path))
        self.classifier.eval()

        return epoch, loss


if __name__ == "__main__":
    parser = ArgumentParser(description='Train image neural net')
    parser.add_argument('--audio_dir', '-a', type=str, default='audio', help='Dir containing audio spectograms for classification')
    parser.add_argument('--model_name', '-m', type=str, help='Name of the model for checkpointing')
    parser.add_argument('--batch_size', '-b', type=int, help="Number of images per batch")
    parser.add_argument('--epochs', '-e', type=int, help="Number of training epochs")
    args = parser.parse_args()

    model = AudioNN(args.model_name)
    if not os.path.exists(args.audio_dir):
        print('Provided path', args.audio_dir, 'is not a valid directory. Please try again')
        exit(-1)
    dataloader = model.load_data(args.audio_dir, args.batch_size)
    audio_features = model.get_features(dataloader)

    model.train_with_features(audio_features, batch_size=args.batch_size, num_epochs=args.epochs)


