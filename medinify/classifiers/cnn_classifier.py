
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
from itertools import zip_longest


class Grouper:
    def __init__(self, data1, data2, n):
        assert data1.shape[0] == data1.shape[0]
        self.data1 = data1
        self.data2 = data2
        self.n = n
        self.first_index, self.last_index = -1, -1
        if len(data1) > 0:
            self.first_index = 0
        if len(data1) > n - 1:
            self.last_index = n

    def __iter__(self):
        return self

    def __next__(self):
        if self.last_index == -1:
            chunk1 = self.data1[self.first_index:]
            chunk2 = self.data2[self.first_index:]
        else:
            chunk1 = self.data1[self.first_index:self.last_index]
            chunk2 = self.data2[self.first_index:self.last_index]
        if self.first_index == -1:
            raise StopIteration
        self.first_index = self.last_index
        if self.last_index + self.n < len(self.data1):
            self.last_index += self.n
        else:
            self.last_index = -1
        return chunk1, chunk2


class CNNLearner:

    default_representation = 'matrix'

    def __init__(self):
        self.network = None

    def fit(self, features, labels, model, n_epochs=10):
        network = ClassificationNetwork(model.processor)
        optimizer = optim.Adam(network.parameters(), lr=0.001)
        criterion = nn.BCEWithLogitsLoss()
        network.train()

        for i, epoch in enumerate(range(1, n_epochs + 1)):
            print('\nStarting Epoch ' + str(epoch))
            epoch_losses = []

            for feature_batch, label_batch in Grouper(features, labels, n=25):
                # TODO: Add Padding Indices
                """
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
                """

        """
            average_loss = sum(epoch_losses) / len(epoch_losses)
            print('Epoch {} average loss: {:.4f}'.format(epoch, average_loss))

        self.network = network
        """

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

    def __init__(self, processor):
        super(ClassificationNetwork, self).__init__()

        self.embed_words = nn.Embedding(len(processor.index_to_word), processor.w2v.vector_size)
        lookup_table = torch.tensor(processor.get_lookup_table(), dtype=torch.float64)
        self.embed_words.weight = nn.Parameter(lookup_table)

        self.conv1 = nn.Conv1d(in_channels=100, out_channels=100, kernel_size=2).double()
        self.conv2 = nn.Conv1d(in_channels=100, out_channels=100, kernel_size=3).double()
        self.conv3 = nn.Conv1d(in_channels=100, out_channels=100, kernel_size=4).double()

        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(300, 50).float()
        self.out = nn.Linear(50, 1).float()

    def forward(self, indices):
        """
        Performs forward pass for data batch on CNN
        """
        indices = torch.tensor(indices, dtype=torch.long)
        embeddings = self.embed_words(indices)
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





