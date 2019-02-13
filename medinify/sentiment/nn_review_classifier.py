"""
Uses a Keras Tensorflow neural network to do sentiment analysis on drug reviews
"""

import csv
import numpy as np
import pandas as pd
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from keras.models import Sequential
from keras.layers import Dense, Dropout
import sklearn.preprocessing as process
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction import DictVectorizer

class NeuralNetReviewClassifier():
    """Uses a Keras Tensorflow neural network to do sentiment analysis on drug reviews

    Attributes:
        negative_threshold (float): Rating cutoff for negative reviews
        positive_threshold (float): Rating cutoff for positive reviews
        stopwords_filename (string): Name of file to be used for stopwords
        model (Sequential): Keras model
    """
    negative_threshold = 2.0
    positive_threshold = 4.0
    model = None

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
            filtered_tokens = {word: True for word in word_tokens if word not in stop_words}

            if float(rating) <= self.negative_threshold:
                negative_comments.append((filtered_tokens, 'neg'))
            if float(rating) >= self.positive_threshold:
                positive_comments.append((filtered_tokens, 'pos'))

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

    @staticmethod # This method is a good candidate for a universal set of functions
    def create_trained_model(train_data, train_target):
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
        fold = 0

        skfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)

        # create a linear stack of layers with an activation function rectified linear unit (relu)
        for train, test in skfold.split(train_data, train_target):
            fold += 1

            model = self.create_trained_model(train_data[train], train_target[train])

            raw_score = model.evaluate(train_data[test], np.array(train_target[test]), verbose=0)
            print("[err, acc] of fold {} : {}".format(fold, raw_score))

            model_scores.append(raw_score[1]*100)

        print(f'Average Accuracy: {np.mean(model_scores)}')
