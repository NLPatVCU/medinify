# Word2Vec
from gensim.models import Word2Vec

import random
import spacy
import time
import ast

import numpy as np
import pandas as pd

# PyTorch
import torch
import torch.nn as nn
from torch.nn import Module
from torch.nn import functional as F
import torch.utils.data
import torch.optim as optim

# TorchText
from torchtext.data import Example, Dataset, Iterator, Field, BucketIterator
from torchtext import data, datasets
from torchtext.vocab import Vectors

class RNNReviewClassifier:
    vectors = None
    embeddings = None
    comment_field = None
    rating_field = None
    loss = nn.BCEWithLogitsLoss()

    def __init__(self, w2v_file):
        """
        Initializes RNNReviewClassifier
        :param w2v_file: embedding file
        """
        vectors = Vectors(w2v_file)
        self.vectors = vectors

    def get_data_loaders(self, train_file, valid_file, batch_size):
        """
        Generates data_loaders given file names
        :param train_file: file with train data
        :param valid_file: file with validation data
        :param batch_size: the loaders' batch sizes
        :return: data loaders
        """
        
        #Reads file into a list
        train_reviews = pd.read_csv(train_file).values.tolist()
        valid_reviews = pd.read_csv(valid_file).values.tolist()

        #Extracting rating from dictionaries
        train_df = pd.read_csv(train_file)
        train_ratings = list(train_df['rating'])
        for index,rating in enumerate(train_ratings):
            train_ratings[index] = ast.literal_eval(rating)
                        
        valid_df = pd.read_csv(valid_file)
        valid_ratings = list(valid_df['rating'])
        for index,rating in enumerate(valid_ratings):
            valid_ratings[index] = ast.literal_eval(rating)
        
        #Assembling data and targets
        train_data = [str(review[0]).lower() for review in train_reviews if review[1] != 3]
        valid_data = [str(review[0]).lower() for review in valid_reviews if review[1] != 3]
        train_target = ['neg' if int(rating["effectiveness"]) in [1, 2] else 'pos' for rating in train_ratings if rating != 3]
        valid_target = ['neg' if int(rating["effectiveness"]) in [1, 2] else 'pos' for rating in valid_ratings if rating != 3]

        return self.generate_data_loaders(train_data, train_target, valid_data, valid_target, batch_size)

    def generate_data_loaders(self, train_data, train_target, valid_data, valid_target, batch_size):
        """
        This function generates TorchText dataloaders for training and validation datasets
        :param train_data: training dataset (list of comment string)
        :param valid_data: validation dataset (list of comment string)
        :param train_target: training data's assosated ratings (list of 'pos' and 'neg')
        :param valid_target: validation data's assosated ratings (list of 'pos' and 'neg')
        :param batch_size: the loaders' batch sizes
        :return: train data loader and validation data loader
        """

        # create TorchText fields
        self.comment_field = data.Field(tokenize = 'spacy', include_lengths = True)
        self.rating_field = data.LabelField(dtype=torch.float)

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
                                       max_size=10000, vectors=self.vectors,
                                       unk_init = torch.Tensor.normal_)
        self.embeddings = self.comment_field.vocab.vectors

        self.rating_field.build_vocab(['pos', 'neg'])
        #Gets index of pad and unk tokens
        PAD_IDX = self.comment_field.vocab.stoi[self.comment_field.pad_token]
        UNK_IDX = self.comment_field.vocab.stoi[self.comment_field.unk_token]
        train_loader = Iterator(train_dataset, batch_size, sort_key=lambda x: len(x))
        valid_loader = Iterator(valid_dataset, batch_size, sort_key=lambda x: len(x))
        
        return train_loader, valid_loader, PAD_IDX, UNK_IDX

    def binary_accuracy(self, preds, y):
        """
        Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
        """

        #round predictions to the closest integer
        rounded_preds = torch.round(torch.sigmoid(preds))
        correct = (rounded_preds == y).float() #convert into float for division 
        acc = correct.sum() / len(correct)
        return acc

    def train(self, network, train_loader):
        """
        Trains a network
        :param network: network being trained
        :param train_loader: train data iterator
        """
        epoch_loss = 0
        epoch_acc = 0
        optimizer = optim.Adam(network.parameters())
        criterion = nn.BCEWithLogitsLoss()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        criterion = criterion.to(device)
        network.train()
        
        for batch in train_loader:
            
            optimizer.zero_grad()
                    
            text, text_lengths = batch.comment
            
            predictions = network(text, text_lengths).squeeze(1)
            
            loss = criterion(predictions, batch.rating)
            
            acc = self.binary_accuracy(predictions, batch.rating)
            
            loss.backward()
            
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            
        return epoch_loss / len(train_loader), epoch_acc / len(train_loader)

    def evaluate_loss_and_accuracy(self, network, valid_loader):
        """
        Evaluates the accuracy of a network with validation data
        :param network: network being evaluated
        :param valid_loader: validation data iterator
        """
        epoch_loss = 0
        epoch_acc = 0
        criterion = nn.BCEWithLogitsLoss()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        criterion = criterion.to(device)
        network.eval()
        
        with torch.no_grad():
        
            for batch in valid_loader:

                text, text_lengths = batch.comment
                
                predictions = network(text, text_lengths).squeeze(1)
                
                loss = criterion(predictions, batch.rating)
                
                acc = self.binary_accuracy(predictions, batch.rating)

                epoch_loss += loss.item()
                epoch_acc += acc.item()
            
        return epoch_loss / len(valid_loader), epoch_acc / len(valid_loader)
    
    def evaluate(self, network, valid_loader):
        """
        Evaluates the accuracy of a network with validation data
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
        num_batch = 1

        with torch.no_grad():
            count = 0
            for batch in valid_loader:
                text, text_lengths = batch.comment
                predictions = network(text, text_lengths).squeeze(1)
                batch_loss = self.loss(predictions.to(torch.double),
                                        batch.rating.to(torch.double))

                tp, fp, tn, fn = self.batch_metrics(predictions, batch.rating)

                total_tp += tp
                total_tn += tn
                total_fp += fp
                total_fn += fn
                calculated += 1
                total_loss += batch_loss

                num_batch = num_batch + 1

            average_accuracy = ((total_tp + total_tn) * 1.0 / (total_tp + total_tn + total_fp + total_fn)) * 100
            average_precision = (total_tp * 1.0 / (total_tp + total_fp)) * 100
            average_recall = (total_tp * 1.0 / (total_tp + total_fn)) * 100
            average_f1 = 2 * ((average_precision * average_recall) / (average_precision + average_recall))
            print('Evaluation Metrics:')
            print('\nTotal Loss: {}\nAverage Accuracy: {}%\n\nAverage Precision: {}%\nAverage Recall: {}%\nAverage F1: {}%'.format(
                total_loss / len(valid_loader), average_accuracy, average_precision, average_recall, average_f1))
            print('True Positive: {}\tTrue Negative: {}\tFalse Positive: {}\tFalse Negative: {}\n'.format(
                total_tp, total_tn, total_fp, total_fn))

        return average_accuracy, average_precision, average_recall

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

class RNNNetwork(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, 
                 bidirectional, dropout, pad_idx, unk_idx):
        
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        
        self.rnn = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           bidirectional=bidirectional, 
                           dropout=dropout)
        
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

        self.dropout = nn.Dropout(dropout)
        
        #Manually setting seed, can be removed or changed
        SEED = 1234

        torch.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True  
        
        #Accounting for pad and unk tokens
        self.embedding.weight.data[unk_idx] = torch.zeros(embedding_dim)
        self.embedding.weight.data[pad_idx] = torch.zeros(embedding_dim)
        
    def forward(self, text, text_lengths):

        embedded = self.dropout(self.embedding(text))
        
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths, enforce_sorted=False)
        
        packed_output, (hidden, cell) = self.rnn(packed_embedded)
        
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        
        return self.fc(hidden)
    
    



