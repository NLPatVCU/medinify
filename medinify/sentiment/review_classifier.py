"""
Text Classifier primarily for medical text.
Currently can use Naive Bayes, Neural Network, or Decision Tree for sentiment analysis.

Based on work by Amy Olex 11/13/17.
"""

import ast
import random
import nltk
# import keras
import numpy as np
import pandas as pd
from sklearn import svm
import nltk.classify.util
from medinify.sentiment.nn_review_classifier import NeuralNetReviewClassifier
# from keras.models import Sequential
# from keras.layers import Dense, Dropout
from nltk.classify import NaiveBayesClassifier
from nltk.classify import DecisionTreeClassifier
import sklearn.preprocessing as process
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction import DictVectorizer
from imblearn.ensemble import BalancedBaggingClassifier


class ReviewClassifier():
    """For performing sentiment analysis on drug reviews

    Attributes:
        attr1 (str): Description of `attr1`.
        attr2 (:obj:`int`, optional): Description of `attr2`.

    """
    classifier_type = None # 'nb', 'dt'
    stop_words_path = None
    iterations = 3
    negative_threshold = 2.0
    positive_threshold = 4.0
    seed = 123

    model = None

    def __init__(self, classifier_type=None, stop_words_path=None):
        self.classifier_type = classifier_type
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

        np.random.seed(self.seed)

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
            

    def train(self, reviews_file):
        """ Trains a classifier based on drug reviews with ratings

        Args:
            reviews_file: Reviews file to use for training.
        """
        ## Parse data from files
        reviews = self.parse_reviews(reviews_file)

        with open('stopwords.txt') as stop_words_file:
            text = self.clean_text(stop_words_file.read())
            stop_words = text.splitlines()

        dataset = self.build_dataset(reviews, stop_words)

        comments = [x[0] for x in dataset]
        ratings = [x[1] for x in dataset]
        kfold = StratifiedKFold(n_splits=self.iterations, shuffle=True, random_state=self.seed)
        cvscores = []

        for train, test in kfold.split(comments, ratings):
            train_data = []
            for item in train:
                train_data.append(dataset[item])

            test_data = []
            for item in test:
                test_data.append(dataset[item])
       
            if self.classifier_type == 'nb':
                self.model = NaiveBayesClassifier.train(train_data)
            elif self.classifier_type == 'dt':
                self.model = DecisionTreeClassifier.train(train_data)

            scores = nltk.classify.util.accuracy(self.model, test_data)
            print("{}%".format(scores * 100))
            cvscores.append(scores * 100)
            # plot_model(model, to_file='model.png')

            if self.classifier_type == 'nb':
                self.model.show_most_informative_features()

            print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))


    def classify(self, comments_file):
        """ Classifies comments as positive or negative based on training.

        Args:
            comments_file: Comments file to classify
        """

        # If model has been trained
        if self.model is not None:

            # Import the file needing classification.
            with open(comments_file) as comments_file:
                comments = comments_file.readlines()

            # Classify each comment and print
            for comment in comments:
                print(str(self.model.classify(self.format_text(comment))) + " :: " + comment)

        # TODO: Decide if we can delete the following
        # parser.add_argument('-d', metavar='domain', type=str,
        #   help='a file with text from a different domain.', required=False, default = None)
        # if args.d is not None:
        #     print("ARGS D")
        #     domain_list = []
        #     with open(args.d) as domainfile:
        #         reader = csv.DictReader(domainfile)
        #         for row in reader:
        #             domain_list.append({'comment': row['comment'], 'rating': row['rating']})
        #     print(str(len(domain_list)))

        #     d_list = []
        #     for c in range(len(domain_list)):
        #         tmp_c = domain_list[c]['comment']
        #         tmp_r = domain_list[c]['rating']

        #         if tmp_r in args.n:
        #             d_list.append((format_text(tmp_c, stop_words), 'neg'))
        #         if tmp_r in args.p:
        #             d_list.append((format_text(tmp_c, stop_words), 'pos'))

        #     # classifier2 = NaiveBayesClassifier.train(domain_list)
        #     model = NaiveBayesClassifier.train(dataset)
        #     domain_accuracy = nltk.classify.util.accuracy(model, d_list)
        #     print('Classifier domain shift accuracy:', domain_accuracy)
