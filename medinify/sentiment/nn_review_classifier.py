'''
Written by Nathan West
VIP Nanoinformatics - Sentiment Classification using Neural Network
10/08/18

This file builds a neural network to analyze reviews of
of clinical drugs, such as citalopram.
'''

import ast
import csv
import string
import numpy as np
import pandas as pd
import nltk
import math
import random
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import CSVLogger
import sklearn.preprocessing as process
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.feature_extraction import DictVectorizer
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn import svm


class NeuralNetReviewClassifier():

    iterations = 3
    negative_threshold = 2.0
    positive_threshold = 4.0
    stopwords_filename = ''
    model = Sequential()

    def __init__(self, stopwords_filename='stopwords.txt'):
        self.stopwords_filename = stopwords_filename

    def remove_stop_words(self, text, stop_words):
        """ Removes all stop words from text.

        Args:
            text: Text to be stripped of stop words.
            stop_words: Stop words to strip from text.
        Returns:
            Stripped text.
        """

        word_list = text.split()
        filtered_words = [word for word in word_list if word not in stop_words]
        text = ' '.join(filtered_words)

        return text

    def clean_text(self, text):
        """ Takes a string, converts to lowercase, and removes all punctuation.

        Args:
            sent: String to be converted.
            stop_words: Stop words to be removed. Defaults to None.
        Returns:
            The cleaned text.
        """

        translator = str.maketrans('', '', string.punctuation)
        text = text.translate(translator).lower()

        return text

    def format_text(self, text, stop_words=None):
        """ Takes a string, converts to lowercase, and removes all punctuation & stop words.

        Args:
            sent: String to be converted.
            stop_words: Stop words to be removed. Defaults to None.
        Returns:
            The formatted text.
        """


        text = self.clean_text(text)

        if stop_words is not None:
            text = self.remove_stop_words(text, stop_words)

        return ({word: True for word in nltk.word_tokenize(text)})

    def parse_reviews(self, reviews_path):
        """ Parses a CSV of reviews into a list of comments with rating.

        Args:
            reviews_file: Reviews file to be parsed.
        Returns:
            The list of comments.
        """

        reviews = []

        with open(reviews_path, newline='') as reviews_file:
            reader = csv.DictReader(reviews_file)

            for row in reader:
                reviews.append({'comment': row['comment'], 'rating': row['rating']})

        return reviews
    
    def build_dataset(self, reviews, stop_words):
        """ Parse and convert positive and negative examples

        Args: 
            reviews: List of reviews with comments and ratings
            stop_words: List of stop words to remove from comments
        Returns:
            A list of important keywords per comment with a positive or negative rating
        """
        positive_comments = []
        negative_comments = []

        for review in reviews:
            comment = review['comment']
            rating = review['rating']

            comment = self.format_text(comment, stop_words)

            if float(rating) <= self.negative_threshold:
                negative_comments.append((comment, 'neg'))
            if float(rating) >= self.positive_threshold:
                positive_comments.append((comment, 'pos'))

        seed = 123
        np.random.seed(seed)

        print("Total Negative Instances:" + str(len(negative_comments)))
        print("Total Positive Instances:" + str(len(positive_comments)))

        negcutoff = math.floor(len(negative_comments) * 1)
        poscutoff = math.floor(len(positive_comments) * 1)

        neg_idx_train = sorted(random.sample(range(len(negative_comments)), negcutoff))
        neg_train = [negative_comments[i] for i in neg_idx_train]

        pos_idx_train = sorted(random.sample(range(len(positive_comments)), poscutoff))
        pos_train = [positive_comments[i] for i in pos_idx_train]

        dataset = neg_train + pos_train
        return dataset

    def vectorize(self, reviews_filename):
        """ Create a vector map from a CSV file of reviews

        Args:
            reviews_filename: Name of CSV file with reviews and ratings
        Returns:
            Vector maps of training data and target
        """
        
        # Open reviews file and put all comments and ratings into a list of dictionaries
        reviews = []
        with open(reviews_filename, newline='') as reviews_file:
            reader = csv.DictReader(reviews_file)

            for row in reader:
                reviews.append({'comment': row['comment'], 'rating': row['rating']})

        # Create list of stopwords from stopwords file
        stopwords = []
        with open(self.stopwords_filename) as stopwords_file:
            text = self.clean_text(stopwords_file.read())
            stopwords = text.splitlines()

        dataset = self.build_dataset(reviews, stopwords)

        # Turn reviews into BOW representation with sentiment rating
        positive_comments = []
        negative_comments = []

        for review in reviews:
            comment = review['comment']
            rating = review['rating']

            comment = self.format_text(comment, stopwords)

            if float(rating) <= self.negative_threshold:
                negative_comments.append((comment, 'neg'))
            if float(rating) >= self.positive_threshold:
                positive_comments.append((comment, 'pos'))

        seed = 123
        np.random.seed(seed)

        print("Total Negative Instances:" + str(len(negative_comments)))
        print("Total Positive Instances:" + str(len(positive_comments)))

        dataset = positive_comments + negative_comments

        # Vectorize the BOW with sentiment reviews
        vectorizer = DictVectorizer(sparse=False)
        data_frame = pd.DataFrame(dataset)
        data_frame.columns = ['data', 'target']

        data = np.array(data_frame['data'])
        train_data = vectorizer.fit_transform(data)

        target = np.array(data_frame['target'])
        encoder = process.LabelEncoder()
        train_target = encoder.fit_transform(target)

        return train_data, train_target

    def create_trained_model(self, train_data, train_target):
        """ Creates and trains new Sequential model
       
        Args:
            train_data: vector map of comments
            train_target: vector map of sentiment based on reviews
        Returns:
            A trained model
        """
        input_dimension = len(train_data[0])

        model = Sequential()
        model.add(Dense(20, input_dim=input_dimension, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(30, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(20, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))

        print('Compiling model...')
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        print('Training model...')
        model.fit(train_data, train_target, epochs=50, verbose=0, class_weight={0: 3, 1: 1})

        print('Model trained!')
        return model

    def train(self, reviews_filename):
        """ Trains a new neural network model with Keras

        Args:
            reviews_filename: CSV file of reviews with ratings
        """
        train_data, train_target = self.vectorize(reviews_filename)
        
        self.model = self.create_trained_model(train_data, train_target)

    def evaluate_average_accuracy(self, reviews_filename):
        """ Use stratified k fold to calculate average accuracy of models

        Args:
            reviews_filename: Filename of CSV with reviews to train on
        """
        train_data, train_target = self.vectorize(reviews_filename)

        model_scores = []
        input_dimension = len(train_data[0])

        skfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)

        fold = 0
        
        # create a linear stack of layers with an activation function rectified linear unit (relu)
        for train, test in skfold.split(train_data, train_target):
            fold += 1

            model = self.create_trained_model(train_data[train], train_target[train])
        
            raw_score = model.evaluate(train_data[test], np.array(train_target[test]), verbose=0)
            print("[err, acc] of fold {} : {}".format(fold, raw_score))

            model_scores.append(raw_score[1]*100)
        
        print(f'Average Accuracy: {np.mean(model_scores)}')
