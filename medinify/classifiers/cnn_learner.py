
from torch.nn import functional as F
from torch.nn import Module
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from torch import optim
from medinify.vectorizers import find_embeddings
from medinify.vectorizers import get_lookup_table
from medinify.classifiers import DataIterator
from gensim.models import KeyedVectors
from tqdm import tqdm


class CNNLearner:
    """
    CNNLearner is used to run fitting and predicting on convolutional
    neural networks (ClassifierNetwork)
    """
    default_representation = 'matrix'

    def __init__(self):
        """
        Constructor for CNNLearner
        :attributes network: (ClassificationNetwork) trained network for predicting
        """
        self.network = None

    def fit(self, features, labels, n_epochs=10):
        """
        Fits a ClassificationNetwork for features and labels
        :param features: (np.array) indices for embedding lookup table
        :param labels: (np.array) numeric representation of labels
        :param n_epochs: number of epochs to train for
        """
        embeddings_file = find_embeddings()
        w2v = KeyedVectors.load_word2vec_format(embeddings_file)
        lookup_table = get_lookup_table(w2v)
        network = ClassificationNetwork(lookup_table)
        optimizer = optim.Adam(network.parameters(), lr=0.001)
        criterion = nn.BCEWithLogitsLoss()
        network.train()

        for i, epoch in enumerate(range(1, n_epochs + 1)):
            print('\nEpoch %d' % (i + 1))
            epoch_losses = []

            for feature_batch, label_batch in tqdm(DataIterator(features, labels, n=25)):
                indices_matrix = self._get_indices_matrix(feature_batch)
                batch_predictions = network(indices_matrix)
                batch_labels = torch.tensor(label_batch.to_numpy(), dtype=torch.float64)
                loss = criterion(batch_predictions, batch_labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss)

            average_loss = sum(epoch_losses) / len(epoch_losses)
            print('Epoch {} average loss: {:.4f}\n'.format(epoch, average_loss))

        print()
        self.network = network

    def predict(self, features, model):
        """
        Predictions labels for features using trained ClassificationNetwork
        :param features: (np.array) indices for embedding lookup table
        :param model: trained model to predict with
        :return: (list[int]) predicted labels
        """
        predictions = []
        with torch.no_grad():
            for feature_batch, label_batch in Grouper(features, np.zeros(features.shape), n=25):
                indices_matrix = self._get_indices_matrix(feature_batch)
                output = model.learner.network(indices_matrix)
                batch_predictions = [
                    tensor.item() for tensor in list(torch.round(torch.sigmoid(output)).to(torch.int64))]
                predictions.extend(batch_predictions)
        return predictions

    @staticmethod
    def _get_indices_matrix(feature_batch):
        """
        Builds np.array matrix from DataFrame of indices arrays
        :param feature_batch: DataFrame of indices arrays
        :return: np.array matrix of indices
        """
        max_len = feature_batch.iloc[-1].shape[0]
        indices_matrix = np.empty((feature_batch.shape[0], max_len), dtype=int)
        for j, indices in enumerate(feature_batch):
            feature_batch = np.pad(indices, (0, max_len - indices.shape[0]))
            indices_matrix[j] = feature_batch
        return indices_matrix


class ClassificationNetwork(Module):
    """
    PyTorch classification convolutional neural network
    """
    def __init__(self, lookup_table):
        """
        Constructs layers of the CNN
        """
        super(ClassificationNetwork, self).__init__()

        self.embed_words = nn.Embedding(lookup_table.shape[0], lookup_table.shape[1])
        lookup_table = torch.tensor(lookup_table, dtype=torch.float64)
        self.embed_words.weight = nn.Parameter(lookup_table)

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=100, out_channels=100, kernel_size=2).double(), nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=100, out_channels=100, kernel_size=3).double(),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=100, out_channels=100, kernel_size=4).double(),
            nn.ReLU()
        )

        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(300, 50).float()
        self.out = nn.Linear(50, 1).float()

    def forward(self, indices):
        """
        Performs forward pass of CNN
        :param indices: tensor of indices to embed
        :return:
        """
        indices = torch.tensor(indices, dtype=torch.long)
        embeddings = self.embed_words(indices)
        embeddings = embeddings.permute(0, 2, 1)

        convolved1 = self.conv1(embeddings)
        convolved2 = self.conv2(embeddings)
        convolved3 = self.conv3(embeddings)

        pooled_1 = F.max_pool1d(convolved1, convolved1.shape[2]).squeeze(2)
        pooled_2 = F.max_pool1d(convolved2, convolved2.shape[2]).squeeze(2)
        pooled_3 = F.max_pool1d(convolved3, convolved3.shape[2]).squeeze(2)

        cat = self.dropout(torch.cat((pooled_1, pooled_2, pooled_3), dim=1)).to(torch.float32)

        linear = self.fc1(cat)
        linear = F.relu(linear)
        return self.out(linear).squeeze(1).to(torch.float64)






