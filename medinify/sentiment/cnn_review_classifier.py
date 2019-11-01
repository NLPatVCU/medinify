
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


class CNNClassifier:

    def fit(self, loader, labels, n_epochs=10):
        # labels is an arbitrary argument, just there for consistency w/ other classifiers

        network = SentimentNetwork(embeddings=loader.dataset.fields['text'].vocab.vectors)
        network.apply(set_weights)

        optimizer = optim.Adam(network.parameters(), lr=0.001)
        criterion = nn.BCEWithLogitsLoss()
        network.train()

        for epoch in range(1, n_epochs + 1):
            print('\nStarting Epoch ' + str(epoch))
            epoch_losses = []

            for batch in loader:
                if batch.text.shape[0] < 4:
                    continue

                predictions = network(batch)
                loss = criterion(predictions, batch.label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss)

            average_loss = sum(epoch_losses) / len(epoch_losses)
            print('Epoch {} average loss: {:.4f}'.format(epoch, average_loss))

        return network

    def evaluate(self, loader, model, verbose=True):
        model.eval()
        criterion = nn.BCEWithLogitsLoss()

        losses = []
        total_confusion_matrix = []

        with torch.no_grad():
            for sample in loader:
                predictions = model(sample)
                sample_loss = criterion(predictions.to(torch.double),
                                        sample.label.to(torch.double))
                losses.append(sample_loss)

                batch_matrix = _batch_metrics(predictions, sample.label)

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

    """
    def validate(self, cnn_dataset, k_folds=10, n_epochs=10):
        skf = StratifiedKFold(n_splits=k_folds)

        accuracies = []
        precisions, recalls, f_measures = {}, {}, {}
        if cnn_dataset.dataset.num_classes == 2:
            precisions = {'Class 1': [], 'Class 2': []}
            recalls = {'Class 1': [], 'Class 2': []}
            f_measures = {'Class 1': [], 'Class 2': []}
        elif cnn_dataset.dataset.num_classes == 3:
            precisions = {'Class 1': [], 'Class 2': [], 'Class 3': []}
            recalls = {'Class 1': [], 'Class 2': [], 'Class 3': []}
            f_measures = {'Class 1': [], 'Class 2': [], 'Class 3': []}
        overall_matrix = None

        for train_indices, test_indices in skf.split(cnn_dataset.dataset.dataset['comment'], cnn_dataset.dataset.dataset['label']):
            train_data = cnn_dataset.dataset.dataset.iloc[train_indices]
            test_data = cnn_dataset.dataset.dataset.iloc[test_indices]
            train_examples = [Example.fromdict(data={'text': train_data.iloc[x]['comment'], 'label': train_data.iloc[x]['label']}, fields=fields) for x in range(train_data.shape[0])]
            test_examples = [Example.fromdict(data={'text': test_data.iloc[x]['comment'], 'label': test_data.iloc[x]['label']}, fields=fields) for x in range(test_data.shape[0])]
            train_dataset = TorchtextDataset(examples=train_examples, fields={'text': text_field, 'label': label_field})
            test_dataset = TorchtextDataset(examples=test_examples, fields={'text': text_field, 'label': label_field})
            vectors = Vectors(cnn_dataset.dataset.word_embeddings_file)
            text_field.build_vocab(train_dataset, vectors=vectors)
            label_field.build_vocab(train_dataset)
            train_loader = BucketIterator(train_dataset, batch_size=10)
            test_loader = BucketIterator(test_dataset, batch_size=10)
            model = self._fit(train_loader, n_epochs=n_epochs, embeddings=vectors.vectors)
            fold_accuracy, fold_precisions, fold_recalls, fold_f_scores, fold_matrix = \
                self._evaluate(test_loader, model)

            accuracies.append(fold_accuracy)
            for i in range(cnn_dataset.dataset.num_classes):
                key_ = 'Class ' + str(i + 1)
                precisions[key_].append(fold_precisions)
                recalls[key_].append(fold_recalls)
                f_measures[key_].append(fold_f_scores)
                if type(overall_matrix) == np.ndarray:
                    overall_matrix += fold_matrix
                else:
                    overall_matrix = fold_matrix

        print_validation_metrics(accuracies, precisions, recalls, f_measures, overall_matrix, cnn_dataset.dataset.num_classes)

        examples = [Example.fromdict(data={'text': self.dataset.dataset.iloc[x]['comment'],
                                           'label': self.dataset.dataset.iloc[x]['label']}, fields=fields)
                    for x in range(self.dataset.dataset.shape[0])]
        cnn_dataset = TorchtextDataset(examples, {'text': self.text_field, 'label': self.label_field})

        for train_indices, test_indices in skf.split(dataset.dataset['comment'], dataset.dataset['label']):
            train_data = dataset.dataset.iloc[train_indices]
            test_data = dataset.dataset.iloc[test_indices]

            train_examples = []
            test_examples = []
        """


class SentimentNetwork(Module):
    """
    A PyTorch Convolutional Neural Network for the sentiment analysis of drug reviews
    """

    def __init__(self, embeddings=None):
        """
        Creates pytorch convnet for training
        :param embeddings: word embeddings
        """
        super(SentimentNetwork, self).__init__()

        self.embed_words = nn.Embedding(len(embeddings), 100)
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
        comments = t.text.permute(1, 0).to(torch.long)
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


def evaluate(evaluation_dataset, trained_model=None):

    """
    if reviews_file and trained_model_file:
        validation_loader, trained_network = setup(reviews_file, w2v_file, trained_model_file=trained_model_file)
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
    """


def validate(input_file, num_folds, w2v_file, rating_type=None, num_epochs=None, batch_size=None):
    """
    if not config.RATING_TYPE:
        config.RATING_TYPE = rating_type
    if not config.EPOCHS:
        config.EPOCHS = num_epochs
    if not config.BATCH_SIZE:
        config.BATCH_SIZE = batch_size

    reviews_list = [review for review in pd.read_csv(input_file).to_numpy()
                    if int(ast.literal_eval(review[1])[config.RATING_TYPE]) != 3]
    comments = [review[0] for review in reviews_list]
    ratings = [review[1] for review in reviews_list]

    skf = KFold(n_splits=num_folds)
    dataset_maker = CNNDataset()

    k_accuracies, k_precisions, k_recalls, k_f_measures = [], [], [], []
    total_confusion_matrix = None
    for train, test in skf.split(comments, ratings):
        train_comments = [comments[x] for x in train]
        train_ratings = [ratings[x] for x in train]
        test_comments = [comments[x] for x in test]
        test_ratings = [ratings[x] for x in test]

        train_loader, test_loader = dataset_maker.get_data_loaders(
            w2v_file=w2v_file, train_comments=train_comments, train_rating=train_ratings,
            validation_comment=test_comments, validation_ratings=test_ratings)

        network = SentimentNetwork(vocab_size=len(dataset_maker.COMMENT.vocab),
                                   embeddings=dataset_maker.COMMENT.vocab.vectors)
        network.apply(set_weights)
        network = fit(w2v_file=w2v_file, train_loader=train_loader, network=network)

        accuracy, precisions, recalls, f_measures, fold_confusion_matrix = evaluate(
            verbose=False, w2v_file=w2v_file, validation_loader=test_loader, trained_network=network)

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
    """


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


def setup(reviews_file, w2v_file, trained_model_file=None):
    data = CNNDataset(config.RATING_TYPE)
    train_loader = data.get_data_loader(review_file=reviews_file, w2v_file=w2v_file)
    if not trained_model_file:
        vocab_size = len(data.COMMENT.vocab.stoi)
        embeddings = data.COMMENT.vocab.vectors
        network = SentimentNetwork(vocab_size, embeddings)
    else:
        state_dict = torch.load(trained_model_file)
        vocab_size = state_dict['embed_words.weight'].shape[0]
        embeddings = state_dict['embed_words.weight']
        network = SentimentNetwork(vocab_size, embeddings)
        network.load_state_dict(state_dict)

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


def print_validation_metrics(accuracies, precisions, recalls, f_measures, overall_matrix, num_classes):
    """
    Prints cross validation metrics
    :param accuracies: list of accuracy scores
    :param precisions: list of precision scores per class
    :param recalls: list of recall scores per class
    :param f_measures: list of f-measure scores per class
    :param overall_matrix: total confusion matrix
    """
    print('\n**********************************************************************\n')
    print('Validation Metrics:')
    print('\n\tAverage Accuracy: {:.4f}% +/- {:.4f}%\n'.format(np.mean(accuracies), np.std(accuracies)))
    for i in range(num_classes):
        key_ = 'Class ' + str(i + 1)
        print('\tClass {} Average Precision: {:.4f}% +/- {:.4f}%'.format(
            i + 1, np.mean(precisions[key_]), np.std(precisions[key_])))
        print('\tClass {} Average Recall: {:.4f}% +/- {:.4f}%'.format(
            i + 1, np.mean(recalls[key_]), np.std(recalls[key_])))
        print('\tClass {} Average F-Measure: {:.4f}% +/- {:.4f}%\n'.format(
            i + 1, np.mean(f_measures[key_]), np.std(f_measures[key_])))
    print('\tOverall Confusion Matrix:\n')
    for row in overall_matrix:
        print('\t{}'.format('\t'.join([str(x) for x in row])))
    print('\n**********************************************************************\n')


