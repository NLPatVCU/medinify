
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from torch.nn import functional as F
from torch.nn import Module
from time import time
import pandas as pd
import ast
import numpy as np
import datetime
import torch
from torchtext.vocab import Vectors
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from torchtext.data import Field, LabelField, BucketIterator, Example
from torchtext.data import Dataset as TorchtextDataset


class CNNLearner:

    def __init__(self):
        self.network = None

    def group(self, iterator, count):
        itr = iter(iterator)
        while True:
            yield tuple([next(itr) for i in range(count)])

    def fit(self, features, labels, n_epochs=10):
        network = ClassificationNetwork()

        optimizer = optim.Adam(network.parameters(), lr=0.001)
        criterion = nn.BCEWithLogitsLoss()
        network.train()

        for i, epoch in enumerate(range(1, n_epochs + 1)):
            print('\nStarting Epoch ' + str(epoch))
            epoch_losses = []

            for matrices in self.group(iter(features), 25):
                max_tokens = 0
                for matrix in matrices:
                    if matrix.shape[0] > max_tokens:
                        max_tokens = matrix.shape[0]
                padded_matrices = np.empty((25, max_tokens, 100))

                if matrix.shape[0] < 4:
                    continue

                prediction = network(matrix)
                label = torch.tensor([labels[i]], dtype=torch.float64)
                loss = criterion(prediction, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss)

            average_loss = sum(epoch_losses) / len(epoch_losses)
            print('Epoch {} average loss: {:.4f}'.format(epoch, average_loss))

        self.network = network

    def predict(self, features):
        predictions = []
        with torch.no_grad():
            for feature in features:
                if feature.shape[0] < 4:
                    predictions.append(1)
                    continue
                output = self.network(feature)
                batch_predictions = [
                    tensor.item() for tensor in list(torch.round(torch.sigmoid(output)).to(torch.int64))]
                predictions.extend(batch_predictions)
        return predictions


class ClassificationNetwork(Module):
    """
    A PyTorch Convolutional Neural Network for the text classification
    """

    def __init__(self, embeddings=None):
        """
        Creates pytorch convnet for training
        :param embeddings: word embeddings
        """
        super(ClassificationNetwork, self).__init__()

        """
        self.embed_words = nn.Embedding(len(embeddings), 100)
        self.embed_words.weight = nn.Parameter(embeddings)
        """

        self.conv1 = nn.Conv1d(in_channels=100, out_channels=100, kernel_size=2).double()
        self.conv2 = nn.Conv1d(in_channels=100, out_channels=100, kernel_size=3).double()
        self.conv3 = nn.Conv1d(in_channels=100, out_channels=100, kernel_size=4).double()

        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(300, 50).float()
        self.out = nn.Linear(50, 1).float()

    def forward(self, t):
        """
        Performs forward pass for data batch on CNN
        """
        embeddings = torch.tensor(t, dtype=torch.float64)
        embeddings = embeddings.permute(1, 0).unsqueeze(0)

        convolved1 = self.conv1(embeddings)
        convolved1 = F.relu(convolved1)

        convolved2 = self.conv2(embeddings)
        convolved2 = F.relu(convolved2)

        convolved3 = self.conv3(embeddings)
        convolved3 = F.relu(convolved3)

        pooled_1 = F.max_pool1d(convolved1, convolved1.shape[2]).squeeze(2)
        pooled_2 = F.max_pool1d(convolved2, convolved2.shape[2]).squeeze(2)
        pooled_3 = F.max_pool1d(convolved3, convolved3.shape[2]).squeeze(2)

        cat = self.dropout(torch.cat((pooled_1, pooled_2, pooled_3), dim=1)).to(torch.float32)

        linear = self.fc1(cat)
        linear = F.relu(linear)
        return self.out(linear).squeeze(1).to(torch.float64)




