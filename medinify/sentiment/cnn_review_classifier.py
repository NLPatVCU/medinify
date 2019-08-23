
from medinify.datasets import CNNDataset
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from torch.nn import functional as F
from torch.nn import Module
from medinify import config
from time import time
import pandas as pd
import ast
import numpy as np
import datetime
import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim


def fit(n_epochs=None, rating_type=None, batch_size=None, reviews_file=None,
        w2v_file=None, train_loader=None, network=None):
    """Trains a CNN network

    :param rating_type: type of ratings to use -> str
    :param batch_size: data loader batch size -> int
    :param n_epochs: number of training epochs -> int
    :param reviews_file: path to file with reviews being trained over -> str
    :param w2v_file: path to word embeddings file -> str
    :param train_loader: training data loader -> torchtext BucketIterator
    :param network: CNN to fit -> SentimentCNN
    """
    if not config.RATING_TYPE:
        config.RATING_TYPE = rating_type
    if not config.BATCH_SIZE:
        config.BATCH_SIZE = batch_size
    if not config.EPOCHS:
        config.EPOCHS = n_epochs

    if reviews_file:
        train_loader, network = setup(reviews_file, w2v_file)
    network.apply(set_weights)

    optimizer = optim.Adam(network.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()

    network.train()

    start_time = time()
    for epoch in range(1, config.EPOCHS + 1):
        epoch_start_time = time()
        print('Starting Epoch ' + str(epoch))
        epoch_losses = []

        for batch in train_loader:
            # if the sentences are shorter than the largest kernel, continue to next batch
            if batch.comment.shape[0] < 4:
                epoch += 1
                continue

            predictions = network(batch)
            loss = criterion(predictions, batch.rating)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss)

        average_loss = sum(epoch_losses) / len(epoch_losses)
        epoch_end_time = time()
        epoch_elapsed = datetime.timedelta(seconds=(epoch_end_time - epoch_start_time))
        print('Epoch {} average loss: {:.4f}'.format(epoch, average_loss))
        print('Epoch {} time elapsed: {}'.format(epoch, epoch_elapsed))

    end_time = time()
    total_elapsed_time = datetime.timedelta(seconds=(end_time - start_time))
    print('\nTotal Elapsed Time: {}'.format(total_elapsed_time))

    return network


def evaluate(batch_size, rating_type, verbose=True, reviews_file=None, w2v_file=None,
             validation_loader=None, trained_model_file=None, trained_network=None):
    """Evaluates the accuracy of a model with validation data

    :param batch_size: data loader batch size -> int
    :param rating_type: type of ratings to use -> str
    :param verbose: whether or not to print evaluation metrics
    :param reviews_file: path to file with reviews being evaluated
    :param w2v_file: path to word embeddings file -> str
    :param validation_loader: data loader -> torchtext BucketIterator
    :param trained_model_file: saved PyTorch model file
    :param trained_network: trained CNN -> SentimentCNN
    """
    if reviews_file and trained_model_file:
        validation_loader, trained_network = setup(reviews_file, w2v_file, batch_size, rating_type)
        trained_network.load_state_dict(torch.load(trained_model_file))

    trained_network.eval()
    criterion = nn.BCEWithLogitsLoss()

    losses = []
    total_confusion_matrix = []

    with torch.no_grad():
        for sample in validation_loader:
            predictions = trained_network(sample)
            sample_loss = criterion(predictions.to(torch.double),
                                    sample.rating.to(torch.double))
            losses.append(sample_loss)

            batch_matrix = _batch_metrics(predictions, sample.rating)

            if type(total_confusion_matrix) == list:
                total_confusion_matrix = batch_matrix
            else:
                total_confusion_matrix += batch_matrix

    tp, fp = total_confusion_matrix[1][1], total_confusion_matrix[1][0]
    tn, fn = total_confusion_matrix[0][0], total_confusion_matrix[0][1]

    accuracy = (tp + tn * 1.0) / (tp + tn + fp + fn) * 100
    precision_1 = ((tp * 1.0) / (tp + fp)) * 100
    precision_2 = ((tn * 1.0) / (tn + fn)) * 100

    recall_1 = ((tp * 1.0) / (tp + fn)) * 100
    recall_2 = ((tn * 1.0) / (tn + fp)) * 100

    f_1 = (2 * ((precision_1 * recall_1) / (precision_1 + recall_1)))
    f_2 = (2 * ((precision_2 * recall_2) / (precision_2 + recall_2)))

    if verbose:
        print('\nEvaluation Metrics:\n')
        print('Accuracy: {:.4f}%'.format(accuracy))
        print('Class 1 Precision: {:.4f}%'.format(precision_1))
        print('Class 1 Recall: {:.4f}%'.format(recall_1))
        print('Class 1 F-Measure: {:.4f}%'.format(f_1))
        print('Class 2 Precision: {:.4f}%'.format(precision_2))
        print('Class 2 Recall: {:.4f}%'.format(recall_2))
        print('Class 2 F-Measure: {:.4f}%\n'.format(f_2))
        print('Confusion Matrix:\n')
        print(total_confusion_matrix)

    precisions = [precision_1, precision_2]
    recalls = [recall_1, recall_2]
    f_measures = [f_1, f_2]
    return accuracy, precisions, recalls, f_measures, total_confusion_matrix


def validate(input_file, num_folds, num_epochs, rating_type, batch=25):
    """
    Evaluates CNN's accuracy using stratified k-fold validation
    :param input_file: dataset file
    :param num_folds: number of k-folds
    :param num_epochs: number of epochs per fold
    :param rating_type: type of ratings to use -> str
    :param batch: batch size -> int
    """
    reviews_list = [review for review in pd.read_csv(input_file).to_numpy()
                    if int(ast.literal_eval(review[1])[rating_type]) != 3]
    comments = [review[0] for review in reviews_list]
    ratings = [review[1] for review in reviews_list]

    skf = KFold(n_splits=num_folds)
    dataset_maker = CNNDataset(rating_type=rating_type)

    k_accuracies, k_precisions, k_recalls, k_f_measures = [], [], [], []
    total_confusion_matrix = None
    for train, test in skf.split(comments, ratings):
        train_comments = [comments[x] for x in train]
        train_ratings = [ratings[x] for x in train]
        test_comments = [comments[x] for x in test]
        test_ratings = [ratings[x] for x in test]

        train_loader, test_loader = dataset_maker.get_data_loaders(
            w2v_file='examples/new_w2v.model', batch_size=5,
            train_comments=train_comments, train_rating=train_ratings,
            validation_comment=test_comments, validation_ratings=test_ratings)

        network = SentimentNetwork(vocab_size=len(dataset_maker.COMMENT.vocab),
                                   embeddings=dataset_maker.COMMENT.vocab.vectors)
        network.apply(set_weights)
        network = fit(batch_size=batch, n_epochs=num_epochs, rating_type=rating_type,
                      train_loader=train_loader, network=network)

        accuracy, precisions, recalls, f_measures, fold_confusion_matrix = evaluate(
            batch_size=batch, rating_type=rating_type, verbose=False,
            validation_loader=test_loader, trained_network=network)

        k_accuracies.append(accuracy)
        k_precisions.append(precisions)
        k_recalls.append(recalls)
        k_f_measures.append(f_measures)
        if type(total_confusion_matrix) != np.ndarray:
            total_confusion_matrix = fold_confusion_matrix
        else:
            total_confusion_matrix += fold_confusion_matrix

    print('\n**********************************************************************\n')
    print('Validation Metrics:')
    print('\n\tAverage Accuracy: {:.4f}% +/- {:.4f}%\n'.format(np.mean(k_accuracies), np.std(k_accuracies)))
    print('\tClass 1 Average Precision: {:.4f}% +/- {:.4f}%'.format(
        np.mean([x[0] for x in k_precisions]), np.std([x[0] for x in k_precisions])))
    print('\tClass 1 Average Recall: {:.4f}% +/- {:.4f}%'.format(
        np.mean([x[0] for x in k_recalls]), np.std([x[0] for x in k_recalls])))
    print('\tClass 1 Average F-Measure: {:.4f}% +/- {:.4f}%\n'.format(
        np.mean([x[0] for x in k_f_measures]), np.std([x[0] for x in k_f_measures])))
    print('\tClass 2 Average Precision: {:.4f}% +/- {:.4f}%'.format(
        np.mean([x[1] for x in k_precisions]), np.std([x[1] for x in k_precisions])))
    print('\tClass 2 Average Recall: {:.4f}% +/- {:.4f}%'.format(
        np.mean([x[1] for x in k_recalls]), np.std([x[1] for x in k_recalls])))
    print('\tClass 2 Average F-Measure: {:.4f}% +/- {:.4f}%\n'.format(
        np.mean([x[1] for x in k_f_measures]), np.std([x[1] for x in k_f_measures])))
    print('\tOverall Confusion Matrix:\n')
    for row in total_confusion_matrix:
        print('\t{}'.format('\t'.join([str(x) for x in row])))
    print('\n**********************************************************************\n')


def save(network, output_file):
    """
    Save a trained network to a file
    :param network: trained network -> SentimentNetwork
    :param output_file: path to output file -> str
    """
    torch.save(network.state_dict(), output_file)


def set_weights(network):
    """
    Randomly initializes weights for neural network
    :param network: network being initialized
    :return: initialized network
    """
    if type(network) == nn.Conv2d or type(network) == nn.Linear:
        torch.nn.init.xavier_uniform_(network.weight)
        network.bias.data.fill_(0.01)

    return network


def setup(reviews_file, w2v_file):
    data = CNNDataset(config.RATING_TYPE)
    train_loader = data.get_data_loader(review_file=reviews_file, w2v_file=w2v_file)
    vocab_size = len(data.COMMENT.vocab.stoi)
    embeddings = data.COMMENT.vocab.vectors
    network = SentimentNetwork(vocab_size, embeddings)

    return train_loader, network


def _batch_metrics(preds, ratings):
    """
    Calculates true positive, false positive, true negative, and false negative
    given a batch's predictions and actual ratings
    :param preds: model predictions
    :param ratings: actual ratings
    :return: confusion matrix
    """
    predictions = torch.round(torch.sigmoid(preds)).to(torch.int64).numpy()
    ratings = ratings.to(torch.int64).numpy()
    matrix = confusion_matrix(ratings, predictions)
    return matrix


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

        self.embed_words = nn.Embedding(vocab_size, 100)
        self.embed_words.weight = nn.Parameter(embeddings)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=100, kernel_size=(2, 100)).double()
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=100, kernel_size=(3, 100)).double()
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=100, kernel_size=(4, 100)).double()

        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(300, 50).float()
        self.out = nn.Linear(50, 1).float()

    def forward(self, t):
        """
        Performs forward pass for data batch on CNN
        """
        comments = t.comment.permute(1, 0).to(torch.long)
        embedded = self.embed_words(comments).unsqueeze(1).to(torch.double)

        convolved1 = self.conv1(embedded).squeeze(3)
        convolved1 = F.relu(convolved1)

        convolved2 = self.conv2(embedded).squeeze(3)
        convolved2 = F.relu(convolved2)

        convolved3 = self.conv3(embedded).squeeze(3)
        convolved3 = F.relu(convolved3)

        pooled_1 = F.max_pool1d(convolved1, convolved1.shape[2]).squeeze(2)
        pooled_2 = F.max_pool1d(convolved2, convolved2.shape[2]).squeeze(2)
        pooled_3 = F.max_pool1d(convolved3, convolved3.shape[2]).squeeze(2)

        cat = self.dropout(torch.cat((pooled_1, pooled_2, pooled_3), dim=1)).to(torch.float32)

        linear = self.fc1(cat)
        linear = F.relu(linear)
        return self.out(linear).squeeze(1).to(torch.float64)

