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

    model = Sequential()

    def __init__(self, stop_words_path):
        self.stop_words_path = stop_words_path

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
         ## Parse and convert positive and negative examples
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

    def nn_train(self, reviews_file):
        
        reviews = self.parse_reviews(reviews_file)
        with open('stopwords.txt') as stop_words_file:
            text = self.clean_text(stop_words_file.read())
            stop_words = text.splitlines()

        dataset = self.build_dataset(reviews, stop_words)

        dv = DictVectorizer(sparse=False)
        data_frame = pd.DataFrame(dataset)
        data_frame.columns = ['data', 'target']

        data = np.array(data_frame['data'])
        train_data = dv.fit_transform(data)

        target = np.array(data_frame['target'])
        enc = process.LabelEncoder()
        train_target = enc.fit_transform(target)

        count = 0
        model_scores = []
        input_dimension = len(train_data[0])
        print(train_data)
        class_weights = {0: 3, 1: 1}

        clf = svm.SVC(gamma='scale')
        bbc = BalancedBaggingClassifier(base_estimator=clf, 
            random_state=20, sampling_strategy='not majority')
        skfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
        count = tot_tn = tot_fn = tot_tp = tot_fp = 0
        
        # create a linear stack of layers with an activation function rectified linear unit (relu)
        for train, test in skfold.split(train_data, train_target):
            
            count += 1
            # model = Sequential()
            self.model.add(Dense(20, input_dim=input_dimension, activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(30, activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(20, activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(1, activation='sigmoid'))

            self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            bbc.fit(train_data[train], train_target[train])
            # print(len(train_data[train]))
            # print(len(train_target[train]))
            # # exit()
            # print(train_target[train])
            # y = np_utils.to_categorical(train_target[train], num_classes=2)
            # print(len(y[0]))
            # print(y)
            self.model.fit(train_data[train], train_target[train], epochs=50,
                batch_size=20, class_weight=class_weights,
                verbose=0)
        
            raw_score = self.model.evaluate(train_data[test], np.array(train_target[test]), verbose=0)
            print("[err, acc] of fold {} : {}".format(count, raw_score))

            model_scores.append(raw_score[1]*100)
        
        print("Average accuracy (train set) - %.2f%%\n" % (np.mean(model_scores)))


    def nn_classify(self, comments_file):
        with open(comments_file) as comments_file:
            comments = comments_file.readlines()

        print(comments)
        # reviews = self.format_text(comments)
        
        # y = np_utils.to_categorical(reviews, num_classes=2)
        # print(y)




