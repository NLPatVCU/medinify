
# Python Libraries
import pickle
import argparse
import json
import datetime
from time import time
import warnings

# Preprocessings
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer

# Classification
from sklearn import svm

# Evaluation
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix

# NN (Currently Unused)
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Dropout


class ReviewClassifier:
    """
    This class is used for the training and evaluation of supervised machine learning classifiers,
    currently including Multinomial Naive Bayes, Random Forest, and Support Vector Machine (all
    implemented using the SciKit Learn library) for the sentiment analysis of online drug reviews

    Attributes:

        classifier_type: str
            acronym for the type of machine learning algorithm used
            ('nb' - Multinomial Naive Bayes, 'rf' - Random Forest, 'svm' - Support Vector Machine)

        model: MultinomialNaiveBayes, RandomForestClassifier, or LinearSVC (depending on classifier type)
            an instance's trained or training classification model

        negative_threshold: float
            star-rating cutoff at with anything <= is labelled negative (default 2.0)

        positive_threshold: float
            star-rating cutoff at with anything >= is labelled positive (default 4.0)

        vectorizer: CountVectorizer
            object for turning dictionary of tokens into numerical representation (vector)
    """

    classifier_type = None
    model = None
    negative_threshold = 2.0
    positive_threshold = 4.0
    vectorizer = None

    def __init__(self, classifier_type=None, negative_threshold=None, positive_threshold=None):
        """
        Initialize an instance of ReviewClassifier for the processing of review data into numerical
        representations, training machine-learning classifiers, and evaluating these classifiers' effectiveness
        :param classifier_type: SciKit Learn supervised machine-learning classifier ('nb', 'svm', or 'rf')
        :param negative_threshold: star-rating cutoff at with anything <= is labelled negative (default 2.0)
        :param positive_threshold: star-rating cutoff at with anything >= is labelled positive (default 4.0)
        """

        self.classifier_type = classifier_type
        self.vectorizer = CountVectorizer()

        if negative_threshold:
            self.negative_threshold = negative_threshold
        if positive_threshold:
            self.positive_threshold = positive_threshold

    def preprocess(self, reviews_filename, remove_stop_words=True):
        """
        Transforms reviews (comments and ratings) into numerical representations (vectors)
        Comments are vectorized into bag-of-words representation
        Ratings are transformed into 0's (negative) and 1's (positive)
        Neutral reviews are discarded

        :param reviews_filename: CSV file with comments and ratings
        :param remove_stop_words: Whether or not to remove stop words
        :return:
        data: list of sparse matrices
            vectorized comments
        target: list of integers
            vectorized ratings
        """

        stop_words = set(stopwords.words('english'))
        tokenizer = RegexpTokenizer(r'\w+')
        df = pd.read_csv(reviews_filename)

        reviews, target = [], []
        num_pos, num_neg, num_neut = 0, 0, 0

        for review in df.values.tolist():
            if type(review[0]) == float:
                continue
            if self.negative_threshold < review[1] < self.positive_threshold:
                num_neut += 1
                continue
            elif review[1] <= self.negative_threshold:
                num_neg += 1
                rating = 0
            else:
                num_pos += 1
                rating = 1
            target.append(rating)
            if remove_stop_words:
                reviews.append(' '.join(word.lower() for word in tokenizer.tokenize(review[0])
                                        if word not in stop_words))
            else:
                reviews.append(' '.join(word.lower() for word in tokenizer.tokenize(review[0])))

        self.vectorizer.fit(reviews)
        data = np.array([self.vectorizer.transform([comment]).toarray() for comment in reviews]).squeeze(1)
        target = np.asarray(target)

        return data, target

    def generate_model(self):
        """
        Creates model based on classifier type
        :return model: untrained machine learning classifier
        """

        model = None

        if self.classifier_type == 'nb':
            model = MultinomialNB(alpha=1, fit_prior=True)
        elif self.classifier_type == 'rf':
            model = RandomForestClassifier(n_estimators=100)
        elif self.classifier_type == 'svm':
            model = svm.LinearSVC(max_iter=10000)

        return model

    def fit(self, data, target):
        """
        Fits model to data and targets
        :param data: list of vectorized comments
        :param target: assosiated ratings (0's and 1's)
        :return model: trained machine learning classifier
        """

        model = self.generate_model()
        model.fit(data, target)
        self.model = model
        return model

    def evaluate_accuracy(self, data, target, model=None, evaluating=False):
        """Evaluate accuracy of current model on new data

        Args:
            data: vectorized comments for feed into model
            target: actual ratings assosiated with data
            model: trained model to evaluate (if none, the class attribute 'model' will be evaluated)
        """

        if model:
            preds = model.predict(data)
        else:
            preds = self.model.predict(data)

        accuracy, precision1, recall1, f1_1, precision2, recall2, f1_2 = metrics(target, preds)

        if not evaluating:
            print('Evaluation Metrics:')
            print('Accuracy: {}%'.format(accuracy * 100))
            print('Positive Precision: {}%'.format(precision1 * 100))
            print('Positive Recall: {}%'.format(recall1 * 100))
            print('Positive F1-Score: {}%'.format(f1_1 * 100))
            print('Negative Precision: {}%'.format(precision2 * 100))
            print('Negative Recall: {}%'.format(recall2 * 100))
            print('Negative F1-Score: {}%'.format(f1_2 * 100))

        """
        if self.classifier_type == 'nn':
            score = self.model.evaluate(
                test_data, np.array(test_target), verbose=0)[1]
        """

        return accuracy, precision1, recall1, f1_1, precision2, recall2, f1_2

    def evaluate_average_accuracy(self, reviews_filename, n_folds):
        """ Use stratified k fold to calculate average accuracy of models

        Args:
            reviews_filename: Filename of CSV with reviews to train on
            n_folds: int, number of k-folds
        """

        data, target = self.preprocess(reviews_filename)
        splits = StratifiedKFold(n_splits=n_folds)

        accuracies, class_1_precisions, class_1_recalls, class_1_f1s = [], [], [], []
        class_2_precisions, class_2_recalls, class_2_f1s = [], [], []

        for train, test in splits.split(data, target):
            x_train = [data[x] for x in train]
            y_train = [target[x] for x in train]
            x_test = [data[x] for x in test]
            y_test = [target[x] for x in test]

            model = self.generate_model()
            model.fit(x_train, y_train)

            accuracy, precision1, recall1, f1_1, precision2, recall2, f1_2 = self.evaluate_accuracy(x_test,
                                                                                                    y_test,
                                                                                                    model=model,
                                                                                                    evaluating=True)
            accuracies.append(accuracy)
            class_1_precisions.append(precision1)
            class_2_precisions.append(precision2)
            class_1_recalls.append(recall1)
            class_2_recalls.append(recall2)
            class_1_f1s.append(f1_1)
            class_2_f1s.append(f1_2)

        average_accuracy = np.mean(np.array(accuracies)) * 100
        accuracy_std = np.std(accuracies) * 100
        average_precision1 = np.mean(np.array(class_1_precisions)) * 100
        prec1_std = np.std(class_1_precisions) * 100
        average_precision2 = np.mean(np.array(class_2_precisions)) * 100
        prec2_std = np.std(class_2_precisions) * 100
        average_recall1 = np.mean(np.array(class_1_recalls)) * 100
        rec1_std = np.std(class_1_recalls) * 100
        average_recall2 = np.mean(np.array(class_2_recalls)) * 100
        rec2_std = np.std(class_2_recalls) * 100
        average_f1_1 = np.mean(np.array(class_1_f1s)) * 100
        f1_1_std = np.std(class_1_f1s) * 100
        average_f1_2 = np.mean(np.array(class_2_f1s)) * 100
        f1_2_std = np.std(class_2_f1s)

        metrics_ = {'accuracies': accuracies, 'positive_precisions': class_1_precisions,
                    'positive_recalls': class_1_recalls, 'positive_f1_scores': class_1_f1s,
                    'negative_precisions': class_2_precisions, 'negative_recalls': class_2_recalls,
                    'negative_f1_scores': class_2_f1s, 'average_accuracy': average_accuracy,
                    'average_positive_precision': average_precision1, 'average_positive_recall': average_recall1,
                    'average_positive_f1_score': average_f1_1, 'average_negative_precision': average_precision2,
                    'average_negative_recall': average_recall2, 'average_negative_f1_score': average_f1_2}

        print('Validation Metrics:')
        print('Average Accuracy: {}% +/- {}%'.format(average_accuracy, accuracy_std))
        print('Average Class 1 (Positive) Precision: {}% +/- {}%'.format(average_precision1, prec1_std))
        print('Average Class 1 (Positive) Recall: {}% +/- {}%'.format(average_recall1, rec1_std))
        print('Average Class 1 (Positive) F1-Score: {}% +/- {}%'.format(average_f1_1, f1_1_std))
        print('Average Class 2 (Negative) Precision: {}% +/- {}%'.format(average_precision2, prec2_std))
        print('Average Class 2 (Negative) Recall: {}% +/- {}%'.format(average_recall2, rec2_std))
        print('Average Class 2 (Negative) F1-Score: {}% +/- {}%'.format(average_f1_2, f1_2_std))

        return metrics_

    def classify(self, output_file, csv_file=None, text_file=None, evaluate=False):
        """Classifies a list of comments as positive or negative

        Args:
            output_file: txt file to which classification results will output
            csv_file: CSV file with comments to classify
            text_file: txt file with comments and no ratings
            evaluate: whether or not to write evaluation metrics to output file
        """

        if self.model is None:
            raise Exception('Model needs training first')
        if self.model and not self.vectorizer:
            raise Exception('A model must be trained before classifying')
        if text_file and evaluate:
            raise Exception('In order to evaluate the classification, data must be passed in csv format')

        stop_words = set(stopwords.words('english'))
        tokenizer = RegexpTokenizer(r'\w+')
        df = pd.read_csv(csv_file)

        clean_comments = []
        comments = []
        target = []

        for review in df.itertuples():
            if type(review.comment) == float or self.negative_threshold < review.rating < self.positive_threshold:
                continue
            elif review.rating <= self.negative_threshold:
                rating = 0
            else:
                rating = 1
            comments.append(review.comment)
            clean_comments.append(' '.join(word.lower() for word in tokenizer.tokenize(review.comment)
                                           if word not in stop_words))
            target.append(rating)

        data = np.array([self.vectorizer.transform([comment]).toarray() for comment in clean_comments]).squeeze(1)
        predictions = self.model.predict(data)

        classifications_file = open(output_file, 'a')
        for i, comment in enumerate(comments):
            if predictions[i] == 1:
                pred = 'Positive'
            else:
                pred = 'Negative'
            if target[i] == 0:
                actual = 'Negative'
            else:
                actual = 'Positive'
            classifications_file.write('Comment: {}\tPrediction: {}\tActual Rating: {}\n'.format(comment, pred, actual))

        if evaluate:
            accuracy, precision1, recall1, f1_1, precision2, recall2, f1_2 = metrics(target, predictions)
            classifications_file.write('\nEvaluation Metrics:\n')
            classifications_file.write('Accuracy: {}%\nClass 1 (Positive) Precision: {}%\n'
                                       'Class 1 (Positive) Recall: {}%\nClass 1 (Positive) F1-Measure: {}%\n'
                                       'Class 2 (Negative) Precision: {}%\nClass 2 (Negative) Recall: {}%\n'
                                       'Class 2 (Negative) F1-Measure: {}%'.format(accuracy * 100, precision1 * 100,
                                                                                   recall1 * 100, f1_1 * 100,
                                                                                   precision2 * 100, recall2 * 100,
                                                                                   f1_2 * 100))

    def save_model(self, output_file):
        """ Saves a trained model to a file
        """

        with open(output_file, 'wb') as pickle_file:
            pickle.dump(self.model, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.vectorizer, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

        """
        elif self.classifier_type == 'nn':
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
        """

    def load_model(self, model_file=None, tar_file=None, saved_vectorizer=None):
        """ Loads a trained model from a file
        """

        with open(model_file, 'rb') as model_file:
            self.model = pickle.load(model_file)
            self.vectorizer = pickle.load(model_file)

        """
        if saved_vectorizer and tar_file:
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

            os.remove('trained_nn_model.json')
            os.remove('trained_nn_weights.h5')
        """


def metrics(actual_ratings, predicted_ratings):

    matrix = confusion_matrix(actual_ratings, predicted_ratings)
    tn, fp, fn, tp = matrix[0][0], matrix[0, 1], matrix[1, 0], matrix[1][1]
    accuracy = (tp + tn) * 1.0 / (tp + tn + fp + fn)
    precision1, precision2 = (tp * 1.0) / (tp + fp), (tn * 1.0) / (tn + fn)
    recall1, recall2 = (tp * 1.0) / (tp + fn), (tn * 1.0) / (tn + fp)
    f1_1 = 2 * ((precision1 * recall1) / (precision1 + recall1))
    f1_2 = 2 * ((precision2 * recall2) / (precision2 + recall2))

    return accuracy, precision1, recall1, f1_1, precision2, recall2, f1_2


