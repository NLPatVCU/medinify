"""
Text Classifier primarily for medical text.
Currently can use Naive Bayes, Neural Network, or Decision Tree for sentiment analysis.
"""

import csv
import numpy as np
from nltk.classify import NaiveBayesClassifier
from nltk.classify import DecisionTreeClassifier
import nltk.classify.util
import nltk
from sklearn.model_selection import StratifiedKFold
from nltk.corpus import stopwords
from nltk import RegexpTokenizer


class ReviewClassifier():
    """For performing sentiment analysis on drug reviews

    Attributes:
        attr1 (str): Description of `attr1`.
        attr2 (:obj:`int`, optional): Description of `attr2`.

    """
    classifier_type = None # 'nb', 'dt'
    iterations = 10
    negative_threshold = 2.0
    positive_threshold = 4.0
    seed = 123

    model = None

    def __init__(self, classifier_type=None):
        self.classifier_type = classifier_type

    def build_dataset(self, reviews_filename):
        """ Builds dataset of labelled positive and negative reviews

        :param reviews_path: CSV file with comments and ratings
        :return: dataset with labeled positive and negative reviews
        """

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

        return dataset

    def create_trained_model(self, dataset):
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
        return model

    def train(self, reviews_filename):
        """ Trains a new naive bayes model or decision tree model

        Args:
            reviews_filename: CSV file of reviews with ratings
        """
        dataset = self.build_dataset(reviews_filename)

        self.model = self.create_trained_model(dataset)

    def evaluate_average_accuracy(self, reviews_filename):
        """ Use stratified k fold to calculate average accuracy of models

        Args:
            reviews_filename: Filename of CSV with reviews to train on
        """
        dataset = self.build_dataset(reviews_filename)
        comments = [x[0] for x in dataset]
        ratings = [x[1] for x in dataset]

        model_scores = []
        fold = 0

        kfold = StratifiedKFold(n_splits=self.iterations, shuffle=True, random_state=self.seed)

        for train, test in kfold.split(comments, ratings):
            fold += 1

            test_data = []
            train_data = []

            for item in test:
                test_data.append(dataset[item])

            for item in train:
                train_data.append(dataset[item])

            model = self.create_trained_model(train_data)

            raw_score = nltk.classify.util.accuracy(model, test_data)
            print("[err, acc] of fold {} : {}".format(fold, raw_score))

            model_scores.append(raw_score*100)

        print(f'Average Accuracy: {np.mean(model_scores)}')
        return np.mean(model_scores)

