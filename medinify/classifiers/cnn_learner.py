
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from torch import optim
from medinify.vectorizers import find_embeddings
from medinify.vectorizers import get_lookup_table
from medinify.classifiers import CNNClassifier
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
        :attributes network: (CNNClassifier) trained network for predicting
        """
        self.network = None

    def fit(self, features, labels, n_epochs=10):
        """
        Fits a CNNClassifier for features and labels
        :param features: (np.array) indices for embedding lookup table
        :param labels: (np.array) numeric representation of labels
        :param n_epochs: number of epochs to train for
        """
        embeddings_file = find_embeddings()
        w2v = KeyedVectors.load_word2vec_format(embeddings_file)
        lookup_table = get_lookup_table(w2v)
        network = CNNClassifier(lookup_table)
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
        Predictions labels for features using trained CNNClassifier
        :param features: (np.array) indices for embedding lookup table
        :param model: trained model to predict with
        :return: (list[int]) predicted labels
        """
        predictions = []
        with torch.no_grad():
            for feature_batch, label_batch in DataIterator(features, np.zeros(features.shape), n=25):
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







