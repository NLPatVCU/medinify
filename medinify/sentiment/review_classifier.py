"""
Text Classifier primarily for medical text.
Currently can use Naive Bayes, Neural Network, or Decision Tree for sentiment analysis.
"""

import csv
import pickle
from time import time
from datetime import datetime
import numpy as np
import pandas as pd
from nltk.classify import NaiveBayesClassifier
from nltk.classify import DecisionTreeClassifier
import nltk.classify.util
from nltk.corpus import stopwords
from nltk import RegexpTokenizer
from sklearn import svm
from sklearn.model_selection import StratifiedKFold
import sklearn.preprocessing as process
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Dropout
import tarfile
import os
from heapq import nlargest

class ReviewClassifier():
    """For performing sentiment analysis on drug reviews

    Attributes:
        attr1 (str): Description of `attr1`.
        attr2 (:obj:`int`, optional): Description of `attr2`.

    """
    classifier_type = None  # 'nb', 'dt', 'nn', 'rf', 'svm'
    iterations = 10
    negative_threshold = 2.0
    positive_threshold = 4.0
    seed = 123
    vectorizer = None
    encoder = None
    evaluating = False

    model = None

    def __init__(self, classifier_type=None):
        self.classifier_type = classifier_type

    def create_dataset(self, reviews_filename):
        """ Builds dataset of labelled positive and negative reviews

        :param reviews_filename: CSV file with comments and ratings
        :return: dataset with labeled positive and negative reviews
        """

        reviews = []
        with open(reviews_filename, newline='') as reviews_file:
            reader = csv.DictReader(reviews_file)

            for row in reader:
                reviews.append({
                    'comment': row['comment'],
                    'rating': row['rating']
                })

        # Separate reviews based on rating
        positive_comments = []
        negative_comments = []

        for review in reviews:
            comment = review['comment']
            rating = review['rating']

            # Make lowercase
            comment = comment.lower()

            # Remove punctuation and tokenize
            tokenizer = RegexpTokenizer(r'\w+')
            word_tokens = tokenizer.tokenize(comment)

            # Remove stopwords and transform into BOW representation
            stop_words = set(stopwords.words('english'))
            filtered_tokens = {
                word: True for word in word_tokens if word not in stop_words
            }

            if float(rating) <= self.negative_threshold:
                negative_comments.append((filtered_tokens, 'neg'))
            if float(rating) >= self.positive_threshold:
                positive_comments.append((filtered_tokens, 'pos'))

        print("Total Negative Instances:" + str(len(negative_comments)))
        print("Total Positive Instances:" + str(len(positive_comments)))

        dataset = positive_comments + negative_comments

        return dataset

    def split_data_target(self, dataset):
        """Builds train_data and train_target given dataset"""

        # Vectorize the BOW with sentiment reviews
        self.vectorizer = DictVectorizer(sparse=False)
        data_frame = pd.DataFrame(dataset)
        data_frame.columns = ['data', 'target']

        data = np.array(data_frame['data'])
        train_data = self.vectorizer.fit_transform(data)

        target = np.array(data_frame['target'])
        self.encoder = process.LabelEncoder()
        train_target = self.encoder.fit_transform(target)

        return train_data, train_target

    def build_dataset(self, reviews_filename):
        """Given reviews file, builds train_data and train_target datasets"""
        dataset = self.create_dataset(reviews_filename)
        train_data, train_target = self.split_data_target(dataset)
        return train_data, train_target

    def create_trained_model(self,
                             dataset=None,
                             train_data=None,
                             train_target=None):
        """ Creates and trains new model

        Args:
            dataset: dataset with reviews and positive or negative labels
        Returns:
            A trained model
        """

        if self.classifier_type == 'nb':
            model = NaiveBayesClassifier.train(dataset)
        elif self.classifier_type == 'dt':
            model = DecisionTreeClassifier.train(dataset)
        elif self.classifier_type == 'rf':
            forest = RandomForestClassifier(n_estimators=100, random_state=0)
            model = forest.fit(train_data, train_target)
        elif self.classifier_type == 'svm':
            model = svm.LinearSVC(max_iter=10000)
            model.fit(train_data, train_target)
        elif self.classifier_type == 'nn':
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
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])

            print('Training model...')
            model.fit(
                train_data,
                train_target,
                epochs=50,
                verbose=0,
                class_weight={
                    0: 3,
                    1: 1
                })

            print('Model trained!')

        return model

    def train(self, reviews_filename):
        """ Trains a new naive bayes model or decision tree model

        Args:
            reviews_filename: CSV file of reviews with ratings
        """
        if self.classifier_type == 'nb' or self.classifier_type == 'dt':
            dataset = self.create_dataset(reviews_filename)
            self.model = self.create_trained_model(dataset)
        elif self.classifier_type in ['nn', 'rf', 'svm']:
            train_data, train_target = self.build_dataset(reviews_filename)
            self.model = self.create_trained_model(
                train_data=train_data, train_target=train_target)

    def evaluate_average_accuracy(self, reviews_filename):
        """ Use stratified k fold to calculate average accuracy of models

        Args:
            reviews_filename: Filename of CSV with reviews to train on
        """

        dataset = []
        train_data = []
        train_target = []
        comments = []
        ratings = []

        if self.classifier_type == 'nb' or self.classifier_type == 'dt':
            dataset = self.create_dataset(reviews_filename)
            comments = [x[0] for x in dataset]
            ratings = [x[1] for x in dataset]
        elif self.classifier_type in ['nn', 'rf', 'svm']:
            train_data, train_target = self.build_dataset(reviews_filename)

        model_scores = []
        fold = 0

        skfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)

        if self.classifier_type == 'nb' or self.classifier_type == 'dt':
            for train, test in skfold.split(comments, ratings):
                fold += 1

                test_data = []
                train_data = []

                for item in test:
                    test_data.append(dataset[item])

                for item in train:
                    train_data.append(dataset[item])

                model = self.create_trained_model(dataset=train_data)

                scores = nltk.classify.util.accuracy(model, test_data)
                model_scores.append(scores * 100)

                if self.classifier_type == 'nb':
                    model.show_most_informative_features()

                self.log("Accuracy of fold %d : %.2f%%" % (fold, scores * 100))

        elif self.classifier_type in ['nn', 'rf', 'svm']:
            for train, test in skfold.split(train_data, train_target):
                fold += 1

                model = self.create_trained_model(
                    train_data=train_data[train],
                    train_target=train_target[train])

                if self.classifier_type == 'nn':
                    raw_score = model.evaluate(
                        train_data[test], np.array(train_target[test]), verbose=0)
                    self.log("[err, acc] of fold {} : {}".format(fold, raw_score))
                    model_scores.append(raw_score[1] * 100)

                else:
                    raw_score = model.score(train_data[test], train_target[test])
                    model_scores.append(raw_score * 100)
                    self.log("Accuracy of fold %d : %.2f%%" % (fold, raw_score * 100))

        self.log("Final Average Accuracy: %.2f%% (+/- %.2f%%)" % (np.mean(model_scores),
                                               np.std(model_scores)))
        return np.mean(model_scores)

    def save_model(self):
        """ Saves a trained model to a file
        """

        if self.classifier_type == 'nn':
            with open("trained_nn_model.json", "w") as json_file:
                json_file.write(self.model.to_json()) # Save mode
            self.model.save_weights("trained_nn_weights.h5") # Save weights
            with open('trained_nn_vec_encoder.pickle', 'wb') as pickle_file:
                pickle.dump(self.vectorizer, pickle_file)
                pickle.dump(self.encoder, pickle_file)
            tar_file = tarfile.open("trained_nn_model.tar", 'w')
            tar_file.add('trained_nn_model.json')
            tar_file.add('trained_nn_weights.h5')
            tar_file.add('trained_nn_vec_encoder.pickle')
            tar_file.close()

            os.remove('trained_nn_model.json')
            os.remove('trained_nn_weights.h5')
            os.remove('trained_nn_vec_encoder.pickle')

        elif self.classifier_type == 'nb':
            file_name = 'trained_nb_model.pickle'
            with open(file_name, 'wb') as pickle_file:
                pickle.dump(self.model, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

        elif self.classifier_type in ['rf', 'svm']:
            file_name = 'trained_' + self.classifier_type + '_model.pickle'
            with open(file_name, 'wb') as pickle_file:
                pickle.dump(self.model, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(self.vectorizer, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(self.encoder, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

        print("Model has been saved!")

    def load_model(self, pickle_file=None, tar_file=None, json_file=None,
                   h5_file=None, saved_vectorizer=True, file_trained_on=None):
        """ Loads a trained model from a file
        """

        if self.classifier_type in ['nb', 'svm', 'rf']:
            print('Loading model...')
            with open(pickle_file, 'rb') as model_file:
                self.model = pickle.load(model_file)
                if self.classifier_type in ['svm', 'rf']:
                    self.vectorizer = pickle.load(model_file)
                    self.encoder = pickle.load(model_file)
            print('Model loaded!')

        elif saved_vectorizer and tar_file:
            print('Loading model...')
            tfile = tarfile.open(tar_file, 'r')
            tfile.extractall()
            tfile.close()

            with open('trained_nn_model.json', 'r') as json_model:
                loaded_model = json_model.read()
                self.model = model_from_json(loaded_model)

            self.model.load_weights('trained_nn_weights.h5')
            self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            with open('trained_nn_vec_encoder.pickle', 'rb') as pickle_file:
                self.vectorizer = pickle.load(pickle_file)
                self.encoder = pickle.load(pickle_file)

            os.remove('trained_nn_model.json')
            os.remove('trained_nn_weights.h5')
            os.remove('trained_nn_vec_encoder.pickle')

        else:

            dataset = self.create_dataset(file_trained_on)
            self.vectorizer = DictVectorizer(sparse=False)

            data_frame = pd.DataFrame(dataset)
            data_frame.columns = ['data', 'target']

            data = np.array(data_frame['data'])
            self.vectorizer.fit_transform(data)

            target = np.array(data_frame['target'])
            self.encoder = process.LabelEncoder()
            self.encoder.fit_transform(target)

            if self.classifier_type in ['rf', 'svm']:
                if pickle_file:
                    print("Loading model...")
                    with open(pickle_file, 'rb') as pickle_model:
                        self.model = pickle.load(pickle_model)

            elif self.classifier_type == 'nn':
                if json_file and h5_file:
                    print("Loading model...")
                    with open(json_file, 'r') as json_model:
                        loaded_model = json_model.read()
                        self.model = model_from_json(loaded_model)

                    print("Loading model weights...")
                    self.model.load_weights(h5_file)

                    self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        if self.model is not None:
            print("Model has been loaded!")

    def classify(self, output_file, comments_filename=None, comments_text_file=None):
        """Classifies a list of comments as positive or negative

        Args:
            comments_filename: CSV file with comments to classify
        """

        if self.model is None:
            print('Model needs training first')
            return

        comments = []

        if comments_text_file:
            text_file = open(comments_text_file, 'r')
            comments = text_file.readlines()

        else:
            with open(comments_filename, 'r') as csv_file:
                csv_reader = csv.DictReader(csv_file)
                for row in csv_reader: comments.append(row['comment'])

        bow_comments = []

        for comment in comments:
            # Make lowercase
            comment = comment.lower()

            # Remove punctuation and tokenize
            tokenizer = RegexpTokenizer(r'\w+')
            word_tokens = tokenizer.tokenize(comment)

            # Remove stopwords and transform into BOW representation
            stop_words = set(stopwords.words('english'))
            filtered_tokens = {
                word: True for word in word_tokens if word not in stop_words
            }

            bow_comments.append(filtered_tokens)

        classifications_file = open(output_file, 'w')

        if self.classifier_type == 'nb':
            for i in range(len(comments)):
                classifications_file.write(str(self.model.classify(bow_comments[i])) + " :: " + comments[i] + '\n')

        else:
            if self.classifier_type in ['rf', 'svm']:
                for i in range(len(comments)):
                    vectorized_comments = self.vectorizer.transform(bow_comments[i])
                    predict_output = self.model.predict(vectorized_comments)
                    sentiment = ''
                    if predict_output[0] == [0]:
                        sentiment = 'neg'
                    elif predict_output[0] == [1]:
                        sentiment = 'pos'
                    classifications_file.write(sentiment + ' :: ' + comments[i] + '\n')
            elif self.classifier_type == 'nn':
                for i in range(len(comments)):
                    vectorized_comments = self.vectorizer.transform(bow_comments[i])
                    predict_output = self.model.predict_classes(vectorized_comments)
                    sentiment = ''
                    if predict_output[0] == 0:
                        sentiment = 'neg'
                    elif predict_output[0] == 1:
                        sentiment = 'pos'
                    classifications_file.write(sentiment + ' :: ' + comments[i] + '\n')

        print('Classification file written!')

    def log(self, statement):
        """Logs and prints statements
        
        Args:
            statement: Statement to log to file and print
        """
        print(statement)
        with open('output.log', 'a') as output:
            timestamp = time()
            timestamp = datetime.utcfromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
            output.write(timestamp + ' - ' + statement + '\n')

    def evaluate_accuracy(self, test_filename):
        """Evaluate accuracy of current model on new data

        Args:
            test_filename: Filepath of reviews to test on
        """

        score = 0

        if self.classifier_type == 'nb':
            dataset = self.create_dataset(test_filename)
            score = nltk.classify.util.accuracy(self.model, dataset)
            self.model.show_most_informative_features()

        elif self.classifier_type in ['rf', 'svm', 'nn']:

            dataset = self.create_dataset(test_filename)
            data_frame = pd.DataFrame(dataset)
            data_frame.columns = ['data', 'target']

            evaluate_data = np.array(data_frame['data'])
            test_data = self.vectorizer.transform(evaluate_data)

            evaluate_target = np.array(data_frame['target'])
            test_target = self.encoder.fit_transform(evaluate_target)

            if self.classifier_type in ['rf', 'svm']:
                score = self.model.score(test_data, test_target)
            elif self.classifier_type == 'nn':
                score = self.model.evaluate(
                    test_data, np.array(test_target), verbose=0)[1]

        self.log("%s accuracy: %.2f%%" % (self.classifier_type, score * 100))
        return score

    """
    This function is meant to print the most important features for a rf model
    
    def print_top_features(self):
        if not self.model:
            print('Must train a model')
            return
        if self.classifier_type == 'rf':
            word_features = self.vectorizer.feature_names_
            feature_importances = self.model.feature_importances_
            most_important = nlargest(5, feature_importances)

            important_words = []
            importances = feature_importances.tolist()
            for feature in most_important:
                index = importances.index(feature)
                important_word = word_features[index]
                important_words.append(important_word)

            print('Most important words: ' + str(important_words))
    """

