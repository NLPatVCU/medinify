
# Evaluation
from sklearn.model_selection import StratifiedKFold

# Word Embeddings
from gensim.models import Word2Vec

# Medinify
from medinify.sentiment.review_classifier import ReviewClassifier

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

# Misc
import json
import os


class CNNReviewClassifier():
    """For performing sentiment analysis on drug reviews
        Using a PyTorch Convolutional Neural Network

    Attributes:
        embeddings (Word2VecKeyedVectors): wv for word2vec, trained on forums
        encoder - encoder for label tensors
    """

    valid_batch = 25
    train_batch = 25
    vectors = None
    embeddings = None
    model = None
    comment_field = None
    rating_field = None
    optimizer = None
    loss = nn.BCEWithLogitsLoss()

    def __init__(self, w2v_file):

        vectors = Vectors(w2v_file)
        self.vectors = vectors
        self.model = SentimentNetwork(vocab_size=len(vectors.stoi), embeddings=self.embeddings)

    def get_data_loaders(self, train_file, valid_file, batch_size):
        """
        Generates data_loaders given file names
        :param train_file: file with train data
        :param valid_file: file with validation data
        :return: data loaders
        """

        dataset_maker = ReviewClassifier()
        train_data = dataset_maker.create_dataset(train_file)
        valid_data = dataset_maker.create_dataset(valid_file)

        return self.generate_data_loaders(train_data, valid_data, batch_size)

    def generate_data_loaders(self, train_dataset, valid_dataset, batch_size):
        """
        This function generates TorchText dataloaders for training and validation datasets
        :param train_dataset: training dataset
        :param valid_dataset: validation dataset
        :return: train data loader and validation data loader
        """

        # create TorchText fields
        self.comment_field = data.Field(lower=True, dtype=torch.float64)
        rating_field = data.LabelField(dtype=torch.float64)

        # iterate through dataset and generate examples with comment_field and rating_field
        train_examples = []
        valid_examples = []

        for review in train_dataset:
            comment = ' '.join(list(review[0].keys()))
            rating = review[1]
            characters = list(comment)
            review = {'comment': comment, 'characters': characters, 'rating': rating}
            ex = Example.fromdict(data=review,
                                  fields={'comment': ('comment', self.comment_field),
                                          'rating': ('rating', rating_field)})
            train_examples.append(ex)

        for review in valid_dataset:
            comment = ' '.join(list(review[0].keys()))
            rating = review[1]
            characters = list(comment)
            review = {'comment': comment, 'characters': characters, 'rating': rating}
            ex = Example.fromdict(data=review,
                                  fields={'comment': ('comment', self.comment_field),
                                          'rating': ('rating', rating_field)})
            valid_examples.append(ex)

        train_dataset = Dataset(examples=train_examples,
                                fields={'comment': self.comment_field,
                                        'rating': rating_field})
        valid_dataset = Dataset(examples=valid_examples,
                                fields={'comment': self.comment_field,
                                        'rating': rating_field})

        # build comment_field and rating_field vocabularies
        self.comment_field.build_vocab(train_dataset.comment, valid_dataset.comment,
                                       max_size=10000, vectors=self.vectors)
        self.embeddings = self.comment_field.vocab.vectors

        rating_field.build_vocab(['pos', 'neg'])

        # create torchtext iterators for train data and validation data
        train_loader = Iterator(train_dataset, batch_size, sort_key=lambda x: len(x))
        valid_loader = Iterator(valid_dataset, batch_size, sort_key=lambda x: len(x))

        return train_loader, valid_loader

    def batch_confusion_matrix(self, predictions, ratings):
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
        self.train(train_loader, valid_loader, n_epochs)

    def train(self, train_loader, valid_loader, n_epochs):
        """
        Trains network on training data
        :param train_loader: train data iterator
        :param valid_loader: validation loader
        :param n_epochs: number of training epochs
        :return: trained network
        """

        # optimizer for network
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        num_epoch = 1
        for epoch in range(num_epoch, n_epochs + 1):
            print('Starting Epoch ' + str(num_epoch))

            epoch_loss = 0
            total_tp = 0
            total_fp = 0
            total_tn = 0
            total_fn = 0

            calculated = 0

            self.model.train()

            batch_num = 1
            for batch in train_loader:

                if batch_num % 25 == 0:
                    print('On batch ' + str(batch_num) + ' of ' + str(len(train_loader)))

                self.optimizer.zero_grad()

                # if the sentences are shorter than the largest kernel, continue to next batch
                if batch.comment.shape[0] < 4:
                    num_epoch = num_epoch + 1
                    continue

                predictions = self.model(batch).squeeze(1).to(torch.float64)
                tp, fp, tn, fn = self.batch_confusion_matrix(predictions, batch.rating)
                total_tp += tp
                total_tn += tn
                total_fn += fn
                total_fp += fp
                loss = self.loss(predictions, batch.rating)
                loss.backward()
                self.optimizer.step()

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

            self.evaluate(valid_loader)

            num_epoch = num_epoch + 1

    def evaluate(self, valid_loader):
        """
        Evaluates the accuracy of a model with validation data
        :param valid_loader: validation data iterator
        """
        self.model.eval()

        total_loss = 0
        total_tp = 0
        total_tn = 0
        total_fp = 0
        total_fn = 0
        calculated = 0

        num_sample = 1

        with torch.no_grad():

            for sample in valid_loader:

                predictions = self.model(sample).squeeze(1)
                sample_loss = self.loss(predictions.to(torch.double),
                                        sample.rating.to(torch.double))

                tp, fp, tn, fn = self.batch_confusion_matrix(predictions, sample.rating)

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

        return average_accuracy, average_precision, average_recall

    def evaluate_k_fold(self, input_file, num_folds, num_epochs):
        """
        Evaluates CNN's accuracy using stratified k-fold validation
        :param input_file: dataset file
        :param num_folds: number of k-folds
        """
        classifier = ReviewClassifier()
        dataset = classifier.create_dataset(input_file)

        comments = [review[0] for review in dataset]
        ratings = [review[1] for review in dataset]

        skf = StratifiedKFold(n_splits=num_folds)

        total_accuracy = 0
        total_precision = 0
        total_recall = 0

        num_fold = 1
        for train, test in skf.split(comments, ratings):

            print('Fold #' + str(num_fold))
            train_data = [dataset[x] for x in train]
            test_data = [dataset[x] for x in test]

            train_loader, valid_loader = self.generate_data_loaders(train_data, test_data)

            self.train(train_loader, valid_loader, num_epochs)
            fold_accuracy, fold_precision, fold_recall = self.evaluate(valid_loader)
            total_accuracy += fold_accuracy
            total_precision += fold_precision
            total_recall += fold_recall

            num_fold += 1

        average_accuracy = total_accuracy / 5
        average_precision = total_precision / 5
        average_recall = total_recall / 5
        print('Average Accuracy: ' + str(average_accuracy))
        print('Average Precision: ' + str(average_precision))
        print('Average Recall: ' + str(average_recall))

    def train_word_embeddings(self, datasets, output_file, training_epochs):
        """trains word embeddings from data files (csvs)
        Parameters:
            datasets - list of file paths to dataset csvs
            output_file - string with name of w2v file
            training_epochs - number of epochs to train embeddings
        """

        classifier = ReviewClassifier()
        comments = []
        dataset_num = 0
        for csv in datasets:
            dataset_num = dataset_num + 1
            print('Gathering comments from dataset #' + str(dataset_num))
            dataset = classifier.create_dataset(csv)
            dataset_comments = []
            for comment in dataset:
                dataset_comments.append(list(comment[0].keys()))
            print('\nFinished gathering dataset #' + str(dataset_num))
            comments = comments + dataset_comments
        print('\nGenerating Word2Vec model')
        w2v_model = Word2Vec(comments)
        print('Training word embeddings...')
        w2v_model.train(comments, total_examples=len(comments),
                             total_words=len(w2v_model.wv.vocab), epochs=training_epochs)
        print('Finished training!')
        self.word_embeddings = w2v_model.wv
        w2v_model.wv.save_word2vec_format(output_file)


class SentimentNetwork(Module):
    """
    A PyTorch Convolutional Neural Network for the sentiment analysis of drug reviews
    """

    use_w2v = False
    use_c2v = False

    def __init__(self, vocab_size=None, embeddings=None):
        super(SentimentNetwork, self).__init__()

        # embedding layer
        self.embed_words = nn.Embedding(vocab_size, 100, padding_idx=1)
        self.embed_words.weight = nn.Parameter(embeddings)

        # convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=100, kernel_size=(2, 100)).double()  # bigrams
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=100, kernel_size=(3, 100)).double()  # trigrams
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=100, kernel_size=(4, 100)).double()  # 4-grams

        # dropout layer
        self.dropout = nn.Dropout(0.6)

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

