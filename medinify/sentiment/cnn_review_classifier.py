
import sklearn.preprocessing as process
import numpy as np
import random

# Word Embeddings
from gensim.models import Word2Vec, KeyedVectors

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


class CNNReviewClassifier():
    """For performing sentiment analysis on drug reviews
        Using a PyTorch Convolutional Neural Network

    Attributes:
        embeddings (Word2VecKeyedVectors): wv for word2vec, trained on forums
        encoder - encoder for label tensors
    """

    embeddings = None
    comment_field = None
    optimizer = None
    loss = nn.BCEWithLogitsLoss()

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
        self.embeddings = w2v_model.wv
        w2v_model.wv.save_word2vec_format(output_file)

    def generate_data_loaders(self, train_file, valid_file, train_batch_size, valid_batch_size, w2v_file):

        classifier = ReviewClassifier()
        train_data = classifier.create_dataset(train_file)
        valid_data = classifier.create_dataset(valid_file)

        comment_field = data.Field(tokenize='spacy', dtype=torch.float64)
        rating_field = data.LabelField(dtype=torch.float64)

        train_examples = []
        valid_examples = []

        for review in train_data:
            comment = ' '.join(list(review[0].keys()))
            rating = review[1]
            review = {'comment': comment, 'rating': rating}
            ex = Example.fromdict(data=review,
                                  fields={'comment': ('comment', comment_field), 'rating': ('rating', rating_field)})
            train_examples.append(ex)

        for review in valid_data:
            comment = ' '.join(list(review[0].keys()))
            rating = review[1]
            review = {'comment': comment, 'rating': rating}
            ex = Example.fromdict(data=review,
                                  fields={'comment': ('comment', comment_field), 'rating': ('rating', rating_field)})
            valid_examples.append(ex)

        train_dataset = Dataset(examples=train_examples,
                                fields={'comment': comment_field, 'rating': rating_field})
        valid_dataset = Dataset(examples=valid_examples,
                                fields={'comment': comment_field, 'rating': rating_field})

        vectors = Vectors(w2v_file)

        comment_field.build_vocab(train_dataset.comment, valid_dataset.comment, vectors=vectors)
        rating_field.build_vocab(['pos', 'neg'])

        self.embeddings = comment_field.vocab.vectors
        self.comment_field = comment_field

        train_loader = Iterator(train_dataset, train_batch_size, sort_key=lambda x: len(x))
        valid_loader = Iterator(valid_dataset, valid_batch_size, sort_key=lambda x: len(x))

        return train_loader, valid_loader

    def train(self, network, train_loader, n_epochs):

        self.optimizer = torch.optim.Adam(network.parameters(), lr=0.001)

        num_epoch = 1
        for x in range(n_epochs):
            self.loss.zero_grad()

            print('Starting Epoch ' + str(num_epoch))

            epoch_loss = 0

            network.train()

            batch_num = 1
            for batch in train_loader:
                if batch_num % 25 == 0:
                    print('On batch ' + str(batch_num) + ' of ' + str(len(train_loader)))

                self.optimizer.zero_grad()
                if batch.comment.shape[0] < 4:
                    num_epoch = num_epoch + 1
                    continue
                predictions = network(batch.comment).squeeze(1).to(torch.float64)
                loss = self.loss(predictions, batch.rating)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

                batch_num = batch_num + 1

            print('Epoch Loss: ' + str(epoch_loss))
            num_epoch = num_epoch + 1

    def evaluate(self, network, valid_loader):

        network.eval()

        accuracies = []
        num_sample = 1
        with torch.no_grad():

            for sample in valid_loader:

                comments = sample.comment
                ratings = sample.rating.to(torch.float64)
                predictions = network(comments)

                predictions = torch.round(torch.sigmoid(predictions)).to(torch.float64).squeeze(1)

                batch_accuracy = ((torch.eq(predictions, ratings).sum().item() * 1.0) / len(predictions)) * 100

                print('Batch #' + str(num_sample) + ' Accuracy: ' + str(batch_accuracy) + '%')
                accuracies.append(batch_accuracy)

                num_sample = num_sample + 1

            print('Average Accuracy: ' + str(sum(accuracies) / len(accuracies)) + '%')

class SentimentNetwork(Module):
    """
    A PyTorch Convolutional Neural Network for the sentiment analysis of drug reviews
    """

    def __init__(self, vocab_size, embeddings):
        super(SentimentNetwork, self).__init__()

        self.embed = nn.Embedding(vocab_size, 100, padding_idx=1)
        self.embed.weight.data.copy_(embeddings)

        # convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=100, kernel_size=(2, 100)).double()
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=100, kernel_size=(3, 100)).double()
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=100, kernel_size=(4, 100)).double()

        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(3 * 100, 200).float()
        self.fc2 = nn.Linear(200, 100).float()
        self.fc3 = nn.Linear(100, 25).float()
        self.out = nn.Linear(25, 1).float()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, t):
        """
        Performs forward pass on CNN
        """

        # reshape to [1, batch sentence length]
        t = t.permute(1, 0).to(torch.long)

        # turn words into embeddings
        embedded = self.embed(t)

        # reshape to [batch size, sentence length, embed dimension]
        embedded = embedded.permute(0, 1, 2)
        embedded = embedded.unsqueeze(1).to(torch.double)

        # convolve embedded outputs three times
        # to find bigrams, tri-grams, and 4-grams (or different by adjusting kernel sizes)
        convolved1 = self.conv1(embedded).squeeze(3)
        convolved1 = F.relu(convolved1)

        convolved2 = self.conv2(embedded).squeeze(3)
        convolved2 = F.relu(convolved2)
        # print(convolved2.shape[2])

        convolved3 = self.conv3(embedded).squeeze(3)
        convolved3 = F.relu(convolved3)

        pooled_1 = F.max_pool1d(convolved1, convolved1.shape[2]).squeeze(2)
        pooled_2 = F.max_pool1d(convolved2, convolved2.shape[2]).squeeze(2)
        pooled_3 = F.max_pool1d(convolved3, convolved3.shape[2]).squeeze(2)

        cat = self.dropout(torch.cat((pooled_1, pooled_2, pooled_3), dim=1)).to(torch.float32)

        linear = self.fc1(cat)
        linear = F.relu(linear)
        linear = self.fc2(linear)
        linear = F.relu(linear)
        linear = self.fc3(linear)
        linear = F.relu(linear)
        return self.out(linear)

