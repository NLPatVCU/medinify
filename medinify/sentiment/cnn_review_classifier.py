
import sklearn.preprocessing as process
import numpy as np

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

class CNNReviewClassifier():
    """For performing sentiment analysis on drug reviews
        Using a PyTorch Convolutional Neural Network

    Attributes:
        embeddings (Word2VecKeyedVectors): wv for word2vec, trained on forums
        encoder - encoder for label tensors
    """

    embeddings = None
    encoder = None

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

    def load_embeddings(self, embeddings_file):
        """Loads word embeddings from file"""

        self.embeddings = KeyedVectors.load_word2vec_format(embeddings_file)

    def create_data_loader(self, dataset_file):

        """Generates data_loader for CNN
        Parameter:
            dataset_file - csv file with dataset
            w2v_model_path - path to w2v model
        """

        classifier = ReviewClassifier()
        dataset = classifier.create_dataset(dataset_file)
        comments = [list(comment[0].keys()) for comment in dataset]
        ratings = [review[1] for review in dataset]

        averaged_embeddings = []

        for comment in comments:
            comment_tensors = []
            for word in comment:
                if word in list(self.embeddings.vocab):
                    comment_tensors.append(torch.FloatTensor(self.embeddings[word]))
                else:
                    comment_tensors.append(torch.zeros(100))

            comment_tensors = torch.stack(comment_tensors).mean(dim=0, dtype=torch.float64)
            averaged_embeddings.append(comment_tensors)

        train = torch.stack(averaged_embeddings)

        target = np.array(ratings)
        self.encoder = process.LabelEncoder()
        target = self.encoder.fit_transform(target)
        target = torch.tensor(target, dtype=torch.long)
        target.unsqueeze(dim=1)

        train_set = torch.utils.data.TensorDataset(train, target)

        data_loader = torch.utils.data.DataLoader(train_set, batch_size=10)
        return data_loader

    def create_cnn_model(self, train_file, validation_file, n_epochs=10, learning_rating=0.001):
        """
        Creates and trains a CNN
        """

        train_data_loader = self.create_data_loader(train_file)
        valid_data_loader = self.create_data_loader(validation_file)

        network = SentimentNetwork()
        self.train(network, n_epochs, learning_rating, train_data_loader, valid_data_loader)

    def train(self, network, n_epochs, learning_rate, train_loader, valid_loader):
        """
        Trains and validates CNN
        """

        loss = nn.CrossEntropyLoss()
        optimizer = optim.Adam(network.parameters(), lr=learning_rate)

        for epoch in range(n_epochs):

            running_loss = 0.0
            total_train_loss = 0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                inputs = inputs.unsqueeze(dim=1)
                inputs = inputs.unsqueeze(dim=1)

                # zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass, backward pass, optimize
                outputs = network(inputs)
                loss_size = loss(outputs, labels)
                loss_size.backward()
                optimizer.step()

                # Print statistics
                running_loss += loss_size.data.item()
                print('Running loss: ' + str(running_loss))
                running_loss = 0.0
                if epoch % 5 == 0:
                    total_train_loss += loss_size.data.item()
                    print('Total loss: ' + str(total_train_loss))

            print('Finished Training')

        sample = next(iter(valid_loader))
        comments, labels = sample
        comments = comments.unsqueeze(dim=1)
        comments = comments.unsqueeze(dim=1)

        preds = network(comments)
        print(preds)
        print(labels)

class SentimentNetwork(Module):
    """
    A PyTorch Convolutional Neural Network for the sentiment analysis of drug reviews
    """

    def __init__(self):
        super(SentimentNetwork, self).__init__()

        # convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(1, 5)).double()
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=(1, 2)).double()

        # linear layers
        self.fc1 = nn.Linear(in_features=288, out_features=120).double()
        self.fc2 = nn.Linear(in_features=120, out_features=60).double()

        # output layer
        self.out = nn.Linear(in_features=60, out_features=2).double()

    def create_weight_matrix(self):
        """
        Generates a weights matrix from embeddings
        """
        matrix_len = len(self.embeddings.index2word)
        weights_matrix = np.zeros((matrix_len, 100))
        words_found = 0

        for i, word in enumerate(self.embeddings.index2word):
            try:
                weights_matrix[i] = self.embeddings[word]
                words_found += 1
            except KeyError:
                weights_matrix[i] = np.random.normal(scale=0.6, size=100)

        return weights_matrix

    """
    I think this method is unused
    
    def create_emb_layer(self):
        
        weights_matrix = self.create_weight_matrix()
        num_embeddings = len(weights_matrix)
        embedding_dim = len(weights_matrix[0])
        embed = nn.Embedding(num_embeddings, embedding_dim)
        embed.weight = weights_matrix

        return embed, num_embeddings, embedding_dim
    """

    def forward(self, t):
        """
        Performs forward pass on CNN
        """
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        t = t.reshape(-1, 288)
        t = self.fc1(t)
        t = F.relu(t)

        t = self.fc2(t)
        t = F.relu(t)

        t = self.out(t)

        return t
