
import numpy as np
import pandas as pd
import ast

# Evaluation
from sklearn.model_selection import StratifiedKFold

# PyTorch
import torch
import torch.nn as nn
from torch.nn import Module
from torch.nn import functional as F
import torch.utils.data
import torch.optim as optim

# TorchText
from torchtext import data
from torchtext.data import Example, Dataset, Iterator
from torchtext.vocab import Vectors


class CNNReviewClassifier:
    """For performing sentiment analysis on drug reviews
        Using a PyTorch Convolutional Neural Network

    Attributes:
        vectors - TorchText word embedding vectors
        embeddings: torch tensor of word2vec embeddings
        comment_field - TorchText data field for comments
        rating_field - TorchText LabelField for ratings
        loss - CNN loss function
    """

    vectors = None
    embeddings = None
    comment_field = None
    rating_field = None
    loss = nn.BCEWithLogitsLoss()

    def __init__(self, w2v_file):
        """
        Initializes CNNReviewClassifier
        :param w2v_file: embedding file
        """
        vectors = Vectors(w2v_file)
        self.vectors = vectors

    def get_data_loaders(self, train_file, valid_file, batch_size, rating_type):
        """
        Generates data_loaders given file names
        :param train_file: file with train data
        :param valid_file: file with validation data
        :param batch_size: the loaders' batch sizes
        :return: data loaders
        """

        train_reviews = pd.read_csv(train_file).values.tolist()
        valid_reviews = pd.read_csv(valid_file).values.tolist()

        for i, review in enumerate(train_reviews):
            rating_dict = ast.literal_eval(review[1])
            new_rating = int(rating_dict[rating_type])
            train_reviews[i][1] = new_rating

        for i, review in enumerate(valid_reviews):
            rating_dict = ast.literal_eval(review[1])
            new_rating = int(rating_dict[rating_type])
            valid_reviews[i][1] = new_rating

        train_data = [str(review[0]).lower() for review in train_reviews if review[1] != 3]
        valid_data = [str(review[0]).lower() for review in valid_reviews if review[1] != 3]
        train_target = ['neg' if review[1] in [1, 2] else 'pos' for review in train_reviews if review[1] != 3]
        valid_target = ['neg' if review[1] in [1, 2] else 'pos' for review in valid_reviews if review[1] != 3]

        return self.generate_data_loaders(train_data, train_target, valid_data, valid_target, batch_size)

    def generate_data_loaders(self, train_data, train_target, valid_data, valid_target, batch_size):
        """
        This function generates TorchText data-loaders for training and validation datasets
        :param train_data: training dataset (list of comment string)
        :param valid_data: validation dataset (list of comment string)
        :param train_target: training data's associated ratings (list of 'pos' and 'neg')
        :param valid_target: validation data's associated ratings (list of 'pos' and 'neg')
        :param batch_size: the loaders' batch sizes
        :return: train data loader and validation data loader
        """
        # create TorchText fields
        self.comment_field = data.Field(lower=True, dtype=torch.float64)
        self.rating_field = data.LabelField(dtype=torch.float64)

        # iterate through dataset and generate examples with comment_field and rating_field
        train_examples = []
        valid_examples = []

        for i in range(len(train_data)):
            comment = train_data[i]
            rating = train_target[i]
            review = {'comment': comment, 'rating': rating}
            ex = Example.fromdict(data=review,
                                  fields={'comment': ('comment', self.comment_field),
                                          'rating': ('rating', self.rating_field)})
            train_examples.append(ex)

        for i in range(len(valid_data)):
            comment = valid_data[i]
            rating = valid_target[i]
            review = {'comment': comment, 'rating': rating}
            ex = Example.fromdict(data=review,
                                  fields={'comment': ('comment', self.comment_field),
                                          'rating': ('rating', self.rating_field)})
            valid_examples.append(ex)

        train_dataset = Dataset(examples=train_examples,
                                fields={'comment': self.comment_field,
                                        'rating': self.rating_field})
        valid_dataset = Dataset(examples=valid_examples,
                                fields={'comment': self.comment_field,
                                        'rating': self.rating_field})

        # build comment_field and rating_field vocabularies
        self.comment_field.build_vocab(train_dataset.comment, valid_dataset.comment,
                                       max_size=10000, vectors=self.vectors)
        self.embeddings = self.comment_field.vocab.vectors

        self.rating_field.build_vocab(['pos', 'neg'])

        # create torchtext iterators for train data and validation data
        train_loader = Iterator(train_dataset, batch_size, sort_key=lambda x: len(x))
        valid_loader = Iterator(valid_dataset, batch_size, sort_key=lambda x: len(x))

        return train_loader, valid_loader

    def batch_metrics(self, predictions, ratings):
        """
        Calculates true positive, false positive, true negative, and false negative
        given a batch's predictions and actual ratings
        :param predictions: model predictions
        :param ratings: actual ratings
        :return: number of fp, tp, tn, and fn
        """

        rounded_preds = torch.round(torch.sigmoid(predictions))

        preds = rounded_preds.to(torch.int64).numpy()
        ratings = ratings.to(torch.int64).numpy()

        true_pos = 0
        false_pos = 0
        true_neg = 0
        false_neg = 0

        i = 0
        while i < len(preds):
            if preds[i] == 0 and ratings[i] == 0:
                true_neg += 1
            elif preds[i] == 0 and ratings[i] == 1:
                false_neg += 1
            elif preds[i] == 1 and ratings[i] == 1:
                true_pos += 1
            elif preds[i] == 1 and ratings[i] == 0:
                false_pos += 1
            i += 1

        return true_pos, false_pos, true_neg, false_neg

    def train_from_files(self, train_file, valid_file, n_epochs, batch_size):
        """
        Trains a model given train file and validation file
        """

        train_loader, valid_loader = self.get_data_loaders(train_file, valid_file, batch_size)
        network = SentimentNetwork(len(self.vectors.stoi), self.vectors.vectors)
        self.train(network=network, train_loader=train_loader,
                   valid_loader=valid_loader, n_epochs=n_epochs)

    def train(self, network, train_loader, n_epochs, valid_loader=None, evaluate=True):
        """
        Trains network on training data
        :param network: network being trained
        :param train_loader: train data iterator
        :param n_epochs: number of training epochs
        :param valid_loader: validation loader
        :param evaluate: whether or not to evaluate validation set after each epoch
                (set to false during cross-validation)
        """

        optimizer = optim.Adam(network.parameters(), lr=0.001)

        network.train()

        num_epoch = 1
        for epoch in range(num_epoch, n_epochs + 1):
            print('Starting Epoch ' + str(num_epoch))

            epoch_loss = 0
            total_tp = 0
            total_fp = 0
            total_tn = 0
            total_fn = 0

            calculated = 0

            batch_num = 1
            for batch in train_loader:

                if batch_num % 25 == 0:
                    print('On batch ' + str(batch_num) + ' of ' + str(len(train_loader)))

                optimizer.zero_grad()

                # if the sentences are shorter than the largest kernel, continue to next batch
                if batch.comment.shape[0] < 4:
                    num_epoch = num_epoch + 1
                    continue

                predictions = network(batch).squeeze(1).to(torch.float64)
                tp, fp, tn, fn = self.batch_metrics(predictions, batch.rating)
                total_tp += tp
                total_tn += tn
                total_fn += fn
                total_fp += fp
                loss = self.loss(predictions, batch.rating)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                calculated = calculated + 1

                batch_num = batch_num + 1

            epoch_accuracy = (total_tp + total_tn) * 1.0 / (total_tp + total_tn + total_fp + total_fn)
            epoch_precision = total_tp * 1.0 / (total_tp + total_fp)
            epoch_recall = total_tp * 1.0 / (total_tp + total_fn)
            print('\nEpoch Loss: ' + str(epoch_loss / len(train_loader)))
            print('Epoch Accuracy: ' + str(epoch_accuracy * 100) + '%')
            print('Epoch Precision: ' + str(epoch_precision * 100) + '%')
            print('Epoch Recall: ' + str(epoch_recall * 100) + '%')
            print('True Positive: {}\tTrue Negative: {}\tFalse Positive: {}\tFalse Negative: {}\n'.format(
                total_tp, total_tn, total_fp, total_fn))

            if evaluate:
                self.evaluate(network, valid_loader)

            num_epoch = num_epoch + 1

        return network

    def evaluate(self, network, valid_loader):
        """
        Evaluates the accuracy of a model with validation data
        :param network: network being evaluated
        :param valid_loader: validation data iterator
        """

        network.eval()

        total_loss = 0
        total_tp = 0
        total_tn = 0
        total_fp = 0
        total_fn = 0
        calculated = 0

        num_sample = 1

        with torch.no_grad():

            for sample in valid_loader:

                predictions = network(sample).squeeze(1)
                sample_loss = self.loss(predictions.to(torch.double),
                                        sample.rating.to(torch.double))

                tp, fp, tn, fn = self.batch_metrics(predictions, sample.rating)

                total_tp += tp
                total_tn += tn
                total_fp += fp
                total_fn += fn
                calculated += 1
                total_loss += sample_loss

                num_sample = num_sample + 1

            average_accuracy = ((total_tp + total_tn) * 1.0 / (total_tp + total_tn + total_fp + total_fn)) * 100
            average_precision = (total_tp * 1.0 / (total_tp + total_fp)) * 100
            average_recall = (total_tp * 1.0 / (total_tp + total_fn)) * 100
            print('Evaluation Metrics:')
            print('\nTotal Loss: {}\nAverage Accuracy: {}%\n\nAverage Precision: {}%\nAverage Recall: {}%'.format(
                total_loss / len(valid_loader), average_accuracy, average_precision, average_recall))
            print('True Positive: {}\tTrue Negative: {}\tFalse Positive: {}\tFalse Negative: {}\n'.format(
                total_tp, total_tn, total_fp, total_fn))

        return average_accuracy, average_precision, average_recall, total_tp, total_tn, total_fp, total_fn

    def set_weights(self, network):
        """
        Randomly initializes weights for neural network
        :param network: network being initialized
        :return: initialized network
        """
        if type(network) == nn.Conv2d or type(network) == nn.Linear:
            torch.nn.init.xavier_uniform_(network.weight)
            network.bias.data.fill_(0.01)

        return network

    def evaluate_k_fold(self, input_file, num_folds, num_epochs, rating_type):
        """
        Evaluates CNN's accuracy using stratified k-fold validation
        :param input_file: dataset file
        :param num_folds: number of k-folds
        :param num_epochs: number of epochs per fold
        """

        df = pd.read_csv(input_file)
        dataset = df.values.tolist()

        for i, review in enumerate(dataset):
            ratings_dict = ast.literal_eval(review[1])
            new_rating = int(ratings_dict[rating_type])
            dataset[i][1] = new_rating

        comments = [review[0].lower() for review in dataset if review[1] != 3]
        ratings = ['neg' if review[1] in [1, 2] else 'pos' for review in dataset if review[1] != 3]

        skf = StratifiedKFold(n_splits=num_folds)

        accuracies = []
        precisions = []
        recalls = []
        f_measures = []

        total_tp, total_tn, total_fp, total_fn = 0, 0, 0, 0

        for train, test in skf.split(comments, ratings):
            train_data = [comments[x] for x in train]
            train_target = [ratings[x] for x in train]
            test_data = [comments[x] for x in test]
            test_target = [ratings[x] for x in test]

            train_loader, valid_loader = self.generate_data_loaders(train_data, train_target,
                                                                    test_data, test_target, 25)

            network = SentimentNetwork(vocab_size=len(self.comment_field.vocab), embeddings=self.embeddings)

            network.apply(self.set_weights)

            self.train(network, train_loader, num_epochs, evaluate=False)
            fold_accuracy, fold_precision, fold_recall, tp, tn, fp, fn = self.evaluate(network, valid_loader)

            total_tp += tp
            total_tn += tn
            total_fp += fp
            total_fn += fn

            """
            accuracies.append(fold_accuracy)
            precisions.append(fold_precision)
            recalls.append(fold_recall)
            fold_f_measure = 2 * ((fold_precision * fold_recall) / (fold_precision + fold_recall))
            f_measures.append(fold_f_measure)
            """

        """
        average_accuracy = total_accuracy / 5
        average_precision = total_precision / 5
        average_recall = total_recall / 5
        print('Average Accuracy: ' + str(average_accuracy))
        print('Average Precision: ' + str(average_precision))
        print('Average Recall: ' + str(average_recall))
        """

        print('Total True Positive: {}'.format(total_tp))
        print('Total True Negative: {}'.format(total_tn))
        print('Total False Positive: {}'.format(total_fp))
        print('Total False Negative: {}'.format(total_fn))


class SentimentNetwork(Module):
    """
    A PyTorch Convolutional Neural Network for the sentiment analysis of drug reviews
    """

    def __init__(self, vocab_size=None, embeddings=None):
        """
        Creates pytorch convnet for training
        :param vocab_size: size of embedding vocab
        :param embeddings: word embeddings
        """
        super(SentimentNetwork, self).__init__()

        # embedding layer
        self.embed_words = nn.Embedding(vocab_size, 100)
        self.embed_words.weight = nn.Parameter(embeddings)

        # convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=100, kernel_size=(2, 100)).double()  # bigrams
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=100, kernel_size=(3, 100)).double()  # trigrams
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=100, kernel_size=(4, 100)).double()  # 4-grams

        # dropout layer
        self.dropout = nn.Dropout(0.5)

        # fully-connected layers
        self.fc1 = nn.Linear(300, 50).float()
        self.out = nn.Linear(50, 1).float()

    def forward(self, t):
        """
        Performs forward pass for data batch on CNN
        """

        # t starts as batch of shape [sentences length, batch size] with each word
        # represented as integer index
        # reshape to [batch size, sentence length]
        comments = t.comment.permute(1, 0).to(torch.long)
        embedded = self.embed_words(comments).unsqueeze(1).to(torch.double)

        # convolve embedded outputs three times
        # to find bigrams, tri-grams, and 4-grams (or different by adjusting kernel sizes)
        convolved1 = self.conv1(embedded).squeeze(3)
        convolved1 = F.relu(convolved1)

        convolved2 = self.conv2(embedded).squeeze(3)
        convolved2 = F.relu(convolved2)

        convolved3 = self.conv3(embedded).squeeze(3)
        convolved3 = F.relu(convolved3)

        # maxpool convolved outputs
        pooled_1 = F.max_pool1d(convolved1, convolved1.shape[2]).squeeze(2)
        pooled_2 = F.max_pool1d(convolved2, convolved2.shape[2]).squeeze(2)
        pooled_3 = F.max_pool1d(convolved3, convolved3.shape[2]).squeeze(2)

        # concatenate maxpool outputs and dropout
        cat = self.dropout(torch.cat((pooled_1, pooled_2, pooled_3), dim=1)).to(torch.float32)

        # fully connected layers
        linear = self.fc1(cat)
        linear = F.relu(linear)
        return self.out(linear)

