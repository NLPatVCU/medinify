import torch
from torchtext import data

from torchtext import datasets
from sklearn.preprocessing import OneHotEncoder
from gensim.models import Word2Vec

import random

import torch.nn as nn

import torch.optim as optim

import time

import numpy as np
import pandas as pd



from torch.nn import Module
from torch.nn import functional as F
import torch.utils.data

# TorchText
from torchtext.data import Example, Dataset, Iterator
from torchtext.vocab import Vectors

class RNNReviewClassifier:
    vectors = None
    embeddings = None
    comment_field = None
    rating_field = None
    loss = nn.BCEWithLogitsLoss()
    encoder = OneHotEncoder()

    def get_data_loaders(self, train_file, valid_file, batch_size):
        """
        Generates data_loaders given file names
        :param train_file: file with train data
        :param valid_file: file with validation data
        :param batch_size: the loaders' batch sizes
        :return: data loaders
        """

        train_reviews = pd.read_csv(train_file).values.tolist()
        valid_reviews = pd.read_csv(valid_file).values.tolist()

        train_data = [str(review[0]).lower() for review in train_reviews if review[1] != 3]
        valid_data = [str(review[0]).lower() for review in valid_reviews if review[1] != 3]
        train_target = ['neg' if review[1] in [1, 2] else 'pos' for review in train_reviews if review[1] != 3]
        valid_target = ['neg' if review[1] in [1, 2] else 'pos' for review in valid_reviews if review[1] != 3]

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
                                       max_size=10000, vectors='glove.6B.100d',
                                       unk_init = torch.Tensor.normal_)
        self.embeddings = self.comment_field.vocab.vectors

        self.rating_field.build_vocab(['pos', 'neg'])

        PAD_IDX = self.comment_field.vocab.stoi[self.comment_field.pad_token]
        UNK_IDX = self.comment_field.vocab.stoi[self.comment_field.unk_token]
        # create torchtext iterators for train data and validation data
        train_loader = Iterator(train_dataset, batch_size, sort_key=lambda x: len(x))
        valid_loader = Iterator(valid_dataset, batch_size, sort_key=lambda x: len(x))
        return train_loader, valid_loader, PAD_IDX, UNK_IDX

    #def generate_iterators:
   



    #print(len(train_data))
    #print(len(valid_data))
    #print(len(test_data))


    

    #print(model.embedding.weight.data)

    #print(len(train_data))
    #print(len(test_data))

    #print(vars(train_data.examples[0]))


    """
    INPUT_DIM = len(TEXT.vocab)
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 256
    OUTPUT_DIM = 1
    """

    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    #print(count_parameters(model))


    

    def binary_accuracy(self, preds, y):
        """
        Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
        """

        #round predictions to the closest integer
        rounded_preds = torch.round(torch.sigmoid(preds))
        correct = (rounded_preds == y).float() #convert into float for division 
        acc = correct.sum() / len(correct)
        return acc

    def train(self, model, iterator):
        
        epoch_loss = 0
        epoch_acc = 0
        optimizer = optim.Adam(model.parameters())
        criterion = nn.BCEWithLogitsLoss()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        criterion = criterion.to(device)
        model.train()
        
        for batch in iterator:
            
            optimizer.zero_grad()
                    
            text, text_lengths = batch.comment
            
            predictions = model(text, text_lengths).squeeze(1)
            
            loss = criterion(predictions, batch.rating)
            
            acc = self.binary_accuracy(predictions, batch.rating)
            
            loss.backward()
            
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            
        return epoch_loss / len(iterator), epoch_acc / len(iterator)

    def evaluate(self, model, iterator):
        
        epoch_loss = 0
        epoch_acc = 0
        criterion = nn.BCEWithLogitsLoss()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        criterion = criterion.to(device)
        model.eval()
        
        with torch.no_grad():
        
            for batch in iterator:

                text, text_lengths = batch.comment
                
                predictions = model(text, text_lengths).squeeze(1)
                
                loss = criterion(predictions, batch.rating)
                
                acc = self.binary_accuracy(predictions, batch.rating)

                epoch_loss += loss.item()
                epoch_acc += acc.item()
            
        return epoch_loss / len(iterator), epoch_acc / len(iterator)



    def epoch_time(self, start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs
    def evaluate2(self, network, valid_loader):
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
        num_batch = 1

        with torch.no_grad():

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
        
        SEED = 1234

        torch.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True

        #train_data, valid_data = train_data.split(random_state = random.seed(SEED))

        
        
        #MAX_VOCAB_SIZE = 25000


        #print(TEXT.vocab.freqs.most_common(20))

        #BATCH_SIZE = 64

        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    

        #model = self.to(device)
        
        

        self.embedding.weight.data[unk_idx] = torch.zeros(embedding_dim)
        self.embedding.weight.data[pad_idx] = torch.zeros(embedding_dim)
        
    def forward(self, text, text_lengths):

        #text = [sent len, batch size]
        
        embedded = self.dropout(self.embedding(text))
        
        #embedded = [sent len, batch size, emb dim]
        #find out about enforce!!!
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths, enforce_sorted=False)
        
        packed_output, (hidden, cell) = self.rnn(packed_embedded)
        
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        
        #output = [sent len, batch size, hid dim]
        #hidden = [1, batch size, hid dim]
        
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        
        return self.fc(hidden)
    
    




    
"""
N_EPOCHS = 5

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

    start_time = time.time()
    
    train_loss, train_acc = train(model, train_iterator)
    valid_loss, valid_acc = evaluate(model, valid_iterator)
    
    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut2-model.pt')
    
    print(str(epoch) + " " + str(epoch_mins) + " " + str(epoch_secs))
    print(str(train_loss) + " " + str(train_acc * 100) + "%")
    print(str(valid_loss) + " " + str(valid_acc * 100) + "%")

model.load_state_dict(torch.load('tut2-model.pt'))

test_loss, test_acc = evaluate(model, test_iterator)

print(str(test_loss) + " " + str(test_acc))
"""