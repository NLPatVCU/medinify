
# Python Libraries
import pickle
import argparse
import json
import datetime
from time import time
import warnings
import tarfile
import os

# Preprocessings
import numpy as np
import pandas as pd
import spacy
from spacy.lang.en import English
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk import RegexpTokenizer
from sklearn.feature_extraction import DictVectorizer

# Classification
from sklearn import svm

# Evaluation
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
# from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV

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

        vectorizer: CountVectorizer or TfidfVectorizer
            object for turning dictionary of tokens into numerical representation (vector)
    """

    classifier_type = None
    model = None
    numclasses = 2
    negative_threshold = 2.0
    positive_threshold = 4.0
    vectorizer = None

    def __init__(self, classifier_type=None, numclasses=2, negative_threshold=None, positive_threshold=None):
        """
        Initialize an instance of ReviewClassifier for the processing of review data into numerical
        representations, training machine-learning classifiers, and evaluating these classifiers' effectiveness
        :param classifier_type: SciKit Learn supervised machine-learning classifier ('nb', 'svm', or 'rf')
        :param negative_threshold: star-rating cutoff at with anything <= is labelled negative (default 2.0)
        :param positive_threshold: star-rating cutoff at with anything >= is labelled positive (default 4.0)
        :param use_tfidf: whether or not to set vectorizer to TF-IDF vectorizer (vectorizer
            is default CountVectorizer)
        """

        self.classifier_type = classifier_type
        self.vectorizer = DictVectorizer(sparse=False)
        self.numclasses = numclasses

        if negative_threshold:
            self.negative_threshold = negative_threshold
        if positive_threshold:
            self.positive_threshold = positive_threshold

    def preprocess(self, reviews_filename):
        """
        Transforms reviews (comments and ratings) into numerical representations (vectors)
        Comments are vectorized into bag-of-words representation
        Ratings are transformed into 0's (negative) and 1's (positive)
        Neutral reviews are discarded

        :param reviews_filename: CSV file with comments and ratings
        :return:
        data: list of sparse matrices
            vectorized comments
        target: list of integers
            vectorized ratings
        """

        stop_words = set(stopwords.words('english'))
        sp = spacy.load('en_core_web_sm')

        df = pd.read_csv(reviews_filename)
        raw_data, raw_target = [], []

        for review in df.itertuples():

            if type(review.comment) == float:
                continue
            comment = {token.text: True for token in sp.tokenizer(review.comment.lower()) if token.text
                       not in stop_words}

            if self.numclasses == 2:
                rating = 'pos'
                if review.rating == 3:
                    continue
                if review.rating in [1, 2]:
                    rating = 'neg'
                raw_data.append(comment)
                raw_target.append(rating)

            elif self.numclasses == 3:
                rating = 'neg'
                if review.rating == 3:
                    rating = 'neut'
                elif review.rating in [4, 5]:
                    rating = 'pos'
                raw_data.append(comment)
                raw_target.append(rating)

            elif self.numclasses == 5:
                raw_target.append(review.rating)
                raw_data.append(comment)

        encoder = LabelEncoder()
        target = np.asarray(encoder.fit_transform(raw_target))
        data = self.vectorizer.fit_transform(raw_data)

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

    def evaluate_accuracy(self, data, target, model=None, verbose=False):
        """Evaluate accuracy of current model on new data

        Args:
            data: vectorized comments for feed into model
            target: actual ratings assosiated with data
            model: trained model to evaluate (if none, the class attribute 'model' will be evaluated)
            verbose: Whether or not to print formatted results to console
        """

        if model:
            predictions = model.predict(data)

        else:
            predictions = self.model.predict(data)

        if self.numclasses == 2:
            accuracy, precision1, recall1, f1_1, precision2, recall2, f1_2, tn, fp, fn, tp = self.metrics(target, predictions)
            if verbose:
                print('Evaluation Metrics:')
                print('Accuracy: {}%'.format(accuracy * 100))
                print('Positive Precision: {}%'.format(precision1 * 100))
                print('Positive Recall: {}%'.format(recall1 * 100))
                print('Positive F1-Score: {}%'.format(f1_1 * 100))
                print('Negative Precision: {}%'.format(precision2 * 100))
                print('Negative Recall: {}%'.format(recall2 * 100))
                print('Negative F1-Score: {}%'.format(f1_2 * 100))
        if self.numclasses == 3:
            accuracy, precision1, recall1, f1_1, precision2, recall2, f1_2, precision3, recall3, f1_3, tpPos, tpNeg, tpNeu, \
            fBA, fBC, fAB, fCB, fCA, fAC = self.metrics(target, predictions)
        if self.numclasses == 5:
            accuracy, precision1, recall1, f1_1, precision2, recall2, f1_2, precision3, recall3, f1_3, precision4, recall4, \
            f1_4, precision5, recall5, f1_5, tpOneStar, tpTwoStar, tpThreeStar, tpFourStar, tpFiveStar, fAB, fAC, fAD, fAE, \
            fBA, fBC, fBD, fBE, fCA, fCB, fCD, fCE, fDA, fDB, fDC, fDE, fEA, fEB, fEC, fED = self.metrics(target, predictions)

        elif verbose and self.numclasses == 3:
            print('Evaluation Metrics:')
            print('Accuracy: {}%'.format(accuracy * 100))
            print('Positive Precision: {}%'.format(precision1 * 100))
            print('Positive Recall: {}%'.format(recall1 * 100))
            print('Positive F1-Score: {}%'.format(f1_1 * 100))
            print('Negative Precision: {}%'.format(precision2 * 100))
            print('Negative Recall: {}%'.format(recall2 * 100))
            print('Negative F1-Score: {}%'.format(f1_2 * 100))
            print('Neutral Precision: {}%'.format(precision3 * 100))
            print('Neutral Recall: {}%'.format(recall3 * 100))
            print('Neutral F1-Score: {}%'.format(f1_3 * 100))
        elif verbose and self.numclasses == 5:
            print('Evaluation Metrics:')
            print('Accuracy: {}%'.format(accuracy * 100))
            print('One Star Precision: {}%'.format(precision1 * 100))
            print('One Star Recall: {}%'.format(recall1 * 100))
            print('One Star F1-Score: {}%'.format(f1_1 * 100))
            print('Two Star Precision: {}%'.format(precision2 * 100))
            print('Two Star Recall: {}%'.format(recall2 * 100))
            print('Two Star F1-Score: {}%'.format(f1_2 * 100))
            print('Three Star Precision: {}%'.format(precision3 * 100))
            print('Three Star Recall: {}%'.format(recall3 * 100))
            print('Three Star F1-Score: {}%'.format(f1_3 * 100))
            print('Four Star Precision: {}%'.format(precision4 * 100))
            print('Four Star Recall: {}%'.format(recall4 * 100))
            print('Four Star F1-Score: {}%'.format(f1_4 * 100))
            print('Five Star Precision: {}%'.format(precision5 * 100))
            print('Five Star Recall: {}%'.format(recall5 * 100))
            print('Five Star F1-Score: {}%'.format(f1_5 * 100))

        """
        if self.classifier_type == 'nn':
            score = self.model.evaluate(
                test_data, np.array(test_target), verbose=0)[1]
        """
        if self.numclasses == 2:
            return accuracy, precision1, recall1, f1_1, precision2, recall2, f1_2, tn, fp, fn, tp
        if self.numclasses == 3:
            return accuracy, precision1, recall1, f1_1, precision2, recall2, f1_2, precision3, recall3, \
            f1_3, tpPos, tpNeg, tpNeu, fBA, fBC, fAB, fCB, fCA, fAC
        if self.numclasses == 5:
            return accuracy, precision1, recall1, f1_1, precision2, recall2, f1_2, precision3, recall3, \
            f1_3, precision4, recall4, f1_4, precision5, recall5, f1_5, tpOneStar, tpTwoStar, tpThreeStar, \
            tpFourStar, tpFiveStar, fAB, fAC, fAD, fAE, fBA, fBC, fBD, fBE, fCA, fCB, fCD, fCE, fDA, fDB, \
            fDC, fDE, fEA, fEB, fEC, fED

    def evaluate_average_accuracy(self, reviews_filename, n_folds, verbose=False):
        """ Use stratified k fold to calculate average accuracy of models

        Args:
            reviews_filename: Filename of CSV with reviews to train on
            n_folds: int, number of k-folds
            verbose: Whether or not to print evaluation metrics to console
        """

        data, target = self.preprocess(reviews_filename=reviews_filename)
        splits = StratifiedKFold(n_splits=n_folds)

        if self.numclasses == 2:
            sumtn, sumfp, sumfn, sumtp = 0, 0, 0, 0
            accuracies, class_1_precisions, class_1_recalls, class_1_f1s = [], [], [], []
            class_2_precisions, class_2_recalls, class_2_f1s = [], [], []

        if self.numclasses == 3:
            class_3_precisions, class_3_recalls, class_3_f1s = [], [], []
            sumtpPos, sumtpNeg, sumtpNeu, sumfBA, sumfBC, sumfAB, sumfCB, sumfCA, sumfAC = 0, 0, 0, 0, 0, 0, 0, 0, 0

        if self.numclasses == 5:
            class_3_precisions, class_3_recalls, class_3_f1s = [], [], []
            class_4_precisions, class_4_recalls, class_4_f1s = [], [], []
            class_5_precisions, class_5_recalls, class_5_f1s = [], [], []
            sumtpOneStar, sumtpTwoStar, sumtpThreeStar, sumtpFourStar, sumtpFiveStar, sumfAB, sumfAC, sumfAD, \
            sumfAE, sumfBA, sumfBC, sumfBD, sumfBE, sumfCA, sumfCB, sumfCD, sumfCE, sumfDA, sumfDB, sumfDC, \
            sumfDE, sumfEA, sumfEB, sumfEC, sumfED = 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0

        scores = []
        for train, test in splits.split(data, target):

            x_train = data[train]
            y_train = target[train]
            x_test = data[test]
            y_test = target[test]

            model = self.generate_model()
            model.fit(x_train, y_train)

            if self.numclasses == 2:
                accuracy, precision1, recall1, f1_1, precision2, recall2, f1_2, tn, fp, fn, tp = self.evaluate_accuracy(x_test, y_test, model=model)

                scores.append(model.score(x_test, y_test))
                print(sum(scores)/len(scores))

                sumtn += tn
                sumfn += fn
                sumfp += fp
                sumtp += tp
            if self.numclasses == 3:
                accuracy, precision1, recall1, f1_1, precision2, recall2, f1_2, precision3, recall3, f1_3, tpPos, \
                tpNeg, tpNeu, fBA, fBC, fAB, fCB, fCA, fAC = self.evaluate_accuracy(x_test, y_test, model=model)
                sumtpPos += tpPos
                sumtpNeg += tpNeg
                sumtpNeu += tpNeu
                sumfBA += fBA
                sumfBC += fBC
                sumfAB += fAB
                sumfCB += fCB
                sumfCA += fCA
                sumfAC += fAC
            if self.numclasses == 5:
                accuracy, precision1, recall1, f1_1, precision2, recall2, f1_2, precision3, recall3, f1_3, precision4, \
                recall4, f1_4, precision5, recall5, f1_5, tpOneStar, tpTwoStar, tpThreeStar, tpFourStar, tpFiveStar, \
                fAB, fAC, fAD, fAE, fBA, fBC, fBD, fBE, fCA, fCB, fCD, fCE, fDA, fDB, fDC, fDE, fEA, fEB, fEC, fED \
                = self.evaluate_accuracy(x_test, y_test, model=model)
                sumtpOneStar += tpOneStar
                sumtpTwoStar += tpTwoStar
                sumtpThreeStar += tpThreeStar
                sumtpFourStar += tpFourStar
                sumtpFiveStar += tpFiveStar 
                sumfAB += fAB
                sumfAC += fAC
                sumfAD += fAD
                sumfAE += fAE
                sumfBA += fBA
                sumfBC += fBC
                sumfBD += fBD
                sumfBE += fBE
                sumfCA += fCA
                sumfCB += fCB
                sumfCD += fCD
                sumfCE += fCE
                sumfDA += fDA
                sumfDB += fDB
                sumfDC += fDC
                sumfDE += fDE
                sumfEA += fEA
                sumfEB += fEB
                sumfEC += fEC
                sumfED += fED                                                                                                                                        
            accuracies.append(accuracy)
            class_1_precisions.append(precision1)
            class_2_precisions.append(precision2)
            class_1_recalls.append(recall1)
            class_2_recalls.append(recall2)
            class_1_f1s.append(f1_1)
            class_2_f1s.append(f1_2)
            if self.numclasses == 3 or self.numclasses == 5:
                class_3_precisions.append(precision3)
                class_3_recalls.append(recall3)
                class_3_f1s.append(f1_3)
            if self.numclasses == 5:
                class_4_precisions.append(precision4)
                class_4_recalls.append(recall4)
                class_4_f1s.append(f1_4)
                class_5_precisions.append(precision5)
                class_5_recalls.append(recall5)
                class_5_f1s.append(f1_5)
        average_accuracy = np.mean(np.array(accuracies)) * 100
        average_precision1 = np.mean(np.array(class_1_precisions)) * 100
        average_precision2 = np.mean(np.array(class_2_precisions)) * 100
        average_recall1 = np.mean(np.array(class_1_recalls)) * 100
        average_recall2 = np.mean(np.array(class_2_recalls)) * 100
        average_f1_1 = np.mean(np.array(class_1_f1s)) * 100
        average_f1_2 = np.mean(np.array(class_2_f1s)) * 100
        if self.numclasses == 3 or self.numclasses == 5:
            average_precision3 = np.mean(np.array(class_3_precisions)) * 100
            average_recall3 = np.mean(np.array(class_3_recalls)) * 100
            average_f1_3 = np.mean(np.array(class_3_f1s)) * 100
        if self.numclasses == 5:
            average_precision4 = np.mean(np.array(class_4_precisions)) * 100
            average_recall4 = np.mean(np.array(class_4_recalls)) * 100
            average_f1_4 = np.mean(np.array(class_4_f1s)) * 100
            average_precision5 = np.mean(np.array(class_5_precisions)) * 100
            average_recall5 = np.mean(np.array(class_5_recalls)) * 100
            average_f1_5 = np.mean(np.array(class_5_f1s)) * 100
        if self.numclasses == 2:
            metrics_ = {'accuracies': accuracies, 'positive_precisions': class_1_precisions,
                        'positive_recalls': class_1_recalls, 'positive_f1_scores': class_1_f1s,
                        'negative_precisions': class_2_precisions, 'negative_recalls': class_2_recalls,
                        'negative_f1_scores': class_2_f1s, 'average_accuracy': average_accuracy,
                        'average_positive_precision': average_precision1, 'average_positive_recall': average_recall1,
                        'average_positive_f1_score': average_f1_1, 'average_negative_precision': average_precision2,
                        'average_negative_recall': average_recall2, 'average_negative_f1_score': average_f1_2}
        if self.numclasses == 3:
            metrics_ = {'accuracies': accuracies, 'positive_precisions': class_1_precisions,
                        'positive_recalls': class_1_recalls, 'positive_f1_scores': class_1_f1s,
                        'negative_precisions': class_2_precisions, 'negative_recalls': class_2_recalls,
                        'negative_f1_scores': class_2_f1s, 'average_accuracy': average_accuracy,
                        'average_positive_precision': average_precision1, 'average_positive_recall': average_recall1,
                        'average_positive_f1_score': average_f1_1, 'average_negative_precision': average_precision2,
                        'average_negative_recall': average_recall2, 'average_negative_f1_score': average_f1_2, 
                        'neutral_precisions': class_3_precisions, 'neutral_recalls': class_3_recalls,
                        'neutral_f1_scores': class_3_f1s, 'average_neutral_precision': average_precision3,
                        'average_neutral_recall': average_recall3, 'average_neutral_f1_score': average_f1_3}    
        if self.numclasses == 5:
            metrics_ = {'accuracies': accuracies, 'onestar_precisions': class_1_precisions,
                        'onestar_recalls': class_1_recalls, 'onestar_f1_scores': class_1_f1s,
                        'twostar_precisions': class_2_precisions, 'twostar_recalls': class_2_recalls,
                        'twostar_f1_scores': class_2_f1s, 'average_accuracy': average_accuracy,
                        'average_onestar_precision': average_precision1, 'average_onestar_recall': average_recall1,
                        'average_onestar_f1_score': average_f1_1, 'average_twostar_precision': average_precision2,
                        'average_twostar_recall': average_recall2, 'average_twostar_f1_score': average_f1_2, 
                        'threestar_precisions': class_3_precisions, 'threestar_recalls': class_3_recalls, 
                        'threestar_f1_scores': class_3_f1s, 'average_threestar_precision': average_precision3,
                        'average_threestar_recall': average_recall3, 'average_threestar_f1_score': average_f1_3, 
                        'fourstar_precisions': class_4_precisions, 'fourstar_recalls': class_4_recalls, 
                        'fourstar_f1_scores': class_4_f1s, 'average_fourstar_precision': average_precision4,
                        'average_fourstar_recall': average_recall4, 'average_fourstar_f1_score': average_f1_4, 
                        'fivestar_precisions': class_5_precisions, 'fivestar_recalls': class_5_recalls, 
                        'fivestar_f1_scores': class_5_f1s, 'average_fivestar_precision': average_precision5,
                        'average_fivestar_recall': average_recall5, 'average_fivestar_f1_score': average_f1_5}
        if verbose and self.numclasses == 2:
            print('Validation Metrics:')
            print('Average Accuracy: {:.4f}%'.format(average_accuracy))
            print('Average Precision: {:.4f}%'.format((average_precision1 + average_precision2) / 2))
            print('Average Recall: {:.4f}%'.format((average_recall1 + average_recall2) / 2))
            print('Average F1-Score: {:.4f}%'.format((average_f1_1 + average_f1_2) / 2))
            print('Average Class 1 (Positive) Precision: {:.4f}%'.format(average_precision1))
            print('Average Class 1 (Positive) Recall: {:.4f}%'.format(average_recall1))
            print('Average Class 1 (Positive) F1-Score: {:.4f}%'.format(average_f1_1))
            print('Average Class 2 (Negative) Precision: {:.4f}%'.format(average_precision2))
            print('Average Class 2 (Negative) Recall: {:.4f}%'.format(average_recall2))
            print('Average Class 2 (Negative) F1-Score: {:.4f}%'.format(average_f1_2))
            print('Confusion Matrix: ')
            print("\t" + "\t" + "Neg:" + "\t" + "Pos:")
            print("Negative:" + "\t" + str(sumtn) + "\t" + str(sumfp))
            print("Positive:" + "\t" + str(sumfn) + "\t" + str(sumtp))
        if verbose and self.numclasses == 3:
            print('Validation Metrics:')
            print('Average Accuracy: {:.4f}%'.format(average_accuracy))
            print('Average Precision: {:.4f}%'.format((average_precision1 + average_precision2 + average_precision3) / 3))
            print('Average Recall: {:.4f}%'.format((average_recall1 + average_recall2 + average_recall3) / 3))
            print('Average F1-Score: {:.4f}%'.format((average_f1_1 + average_f1_2 + average_f1_3) / 3))
            print('Average Class 1 (Positive) Precision: {:.4f}%'.format(average_precision1))
            print('Average Class 1 (Positive) Recall: {:.4f}%'.format(average_recall1))
            print('Average Class 1 (Positive) F1-Score: {:.4f}%'.format(average_f1_1))
            print('Average Class 2 (Negative) Precision: {:.4f}%'.format(average_precision2))
            print('Average Class 2 (Negative) Recall: {:.4f}%'.format(average_recall2))
            print('Average Class 2 (Negative) F1-Score: {:.4f}%'.format(average_f1_2))
            print('Average Class 3 (Neutral) Precision: {:.4f}%'.format(average_precision3))
            print('Average Class 3 (Neutral) Recall: {:.4f}%'.format(average_recall3))
            print('Average Class 3 (Neutral) F1-Score: {:.4f}%'.format(average_f1_3))
            print('Confusion Matrix: ')
            print("\t" + "\t" + "Neg:" + "\t" + "Neu:" + "\t" + "Pos:")
            print("Negative:" + "\t" + str(sumtpPos) + "\t" + str(sumfAB) + "\t" + str(sumfAC))
            print("Neutral:" + "\t" + str(sumfBA) + "\t" + str(sumtpNeg) + "\t" + str(sumfBC))
            print("Positive:" + "\t" + str(sumfCA) + "\t" + str(sumfCB) + "\t" + str(sumtpNeu))       
        if verbose and self.numclasses == 5:
            print('Validation Metrics:')
            print('Average Accuracy: {:.4f}%'.format(average_accuracy))
            print('Average Precision: {:.4f}%'.format((average_precision1 + average_precision2 + average_precision3 + average_precision4 + average_precision5) / 5))
            print('Average Recall: {:.4f}%'.format((average_recall1 + average_recall2 + average_recall3 + average_recall4 + average_recall5) / 5))
            print('Average F1-Score: {:.4f}%'.format((average_f1_1 + average_f1_2 + average_f1_3 + average_f1_4 + average_f1_5) / 5))
            print('Average One Star Precision: {:.4f}%'.format(average_precision1))
            print('Average One Star Recall: {:.4f}%'.format(average_recall1))
            print('Average One Star F1-Score: {:.4f}%'.format(average_f1_1))
            print('Average Two Star Precision: {:.4f}%'.format(average_precision2))
            print('Average Two Star Recall: {:.4f}%'.format(average_recall2))
            print('Average Two Star F1-Score: {:.4f}%'.format(average_f1_2))
            print('Average Three Star Precision: {:.4f}%'.format(average_precision3))
            print('Average Three Star Recall: {:.4f}%'.format(average_recall3))
            print('Average Three Star F1-Score: {:.4f}%'.format(average_f1_3))
            print('Average Four Star Precision: {:.4f}%'.format(average_precision4))
            print('Average Four Star Recall: {:.4f}%'.format(average_recall4))
            print('Average Four Star F1-Score: {:.4f}%'.format(average_f1_4))
            print('Average Five Star Precision: {:.4f}%'.format(average_precision5))
            print('Average Five Star Recall: {:.4f}%'.format(average_recall5))
            print('Average Five Star F1-Score: {:.4f}%'.format(average_f1_5))
            print('Confusion Matrix: ')
            print("\t" + "\t" + "1-Star:" + "\t" + "2-Star:" + "\t" + "3-Star:" + "\t" + "4-Star:" + "\t" + "5-Star:")
            print("One Star:" + "\t" + str(sumtpOneStar) + "\t" + str(sumfAB) + "\t" + str(sumfAC) + "\t" + str(sumfAD) + "\t" + str(sumfAE))
            print("Two Star:" + "\t" + str(sumfBA) + "\t" + str(sumtpTwoStar) + "\t" + str(sumfBC) + "\t" + str(sumfBD) + "\t" + str(sumfBE))
            print("Three Star:" + "\t" + str(sumfCA) + "\t" + str(sumfCB) + "\t" + str(sumtpThreeStar) + "\t" + str(sumfCD) + "\t" + str(sumfCE))
            print("Four Star:" + "\t" + str(sumfDA) + "\t" + str(sumfDB) + "\t" + str(sumfDC) + "\t" + str(sumtpFourStar) + "\t" + str(sumfDE))
            print("Five Star:" + "\t" + str(sumfEA) + "\t" + str(sumfEB) + "\t" + str(sumfEC) + "\t" + str(sumfED) + "\t" + str(sumtpFiveStar))
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

        if self.numclasses == 2:
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
        if self.numclasses == 3:
            for review in df.itertuples():
                if type(review.comment) == float:
                    continue
                elif self.negative_threshold < review.rating < self.positive_threshold:
                    rating = 1
                elif review.rating <= self.negative_threshold:
                    rating = 0
                else:
                    rating = 2
                comments.append(review.comment)
                clean_comments.append(' '.join(word.lower() for word in tokenizer.tokenize(review.comment)
                                            if word not in stop_words))
                target.append(rating)
        if self.numclasses == 5:
            for review in df.itertuples():
                if type(review.comment) == float:
                    continue
                if review.rating == 1.0:
                    rating = 1
                elif review.rating == 2.0:
                    rating = 2
                elif review.rating == 3.0:
                    rating = 3
                elif review.rating == 4.0:
                    rating = 4
                else:
                    rating = 5
                comments.append(review.comment)
                clean_comments.append(' '.join(word.lower() for word in tokenizer.tokenize(review.comment)
                                            if word not in stop_words))
                target.append(rating)

        data = np.array([self.vectorizer.transform([comment]).toarray() for comment in clean_comments])
        predictions = self.model.predict(data)

        classifications_file = open(output_file, 'a')
        if self.numclasses == 2: 
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
        if self.numclasses == 3:
            for i, comment in enumerate(comments):
                if predictions[i] == 2:
                    pred = 'Positive'
                elif predictions[i] == 1:
                    pred = 'Neutral'
                else:
                    pred = 'Negative'
                if target[i] == 0:
                    actual = 'Negative'
                elif target[i] == 1:
                    actual = 'Neutral'
                else:
                    actual = 'Positive'
                classifications_file.write('Comment: {}\tPrediction: {}\tActual Rating: {}\n'.format(comment, pred, actual))
        if self.numclasses == 5:
            for i, comment in enumerate(comments):
                if predictions[i] == 1:
                    pred = 'One Star'
                elif predictions[i] == 2:
                    pred = 'Two Star'
                elif predictions[i] == 3:
                    pred = 'Three Star'
                elif predictions[i] == 4:
                    pred = 'Four Star'
                else:
                    pred = 'Five Star'
                if target[i] == 1:
                    actual = 'One Star'
                elif target[i] == 2:
                    actual = 'Two Star'
                elif target[i] == 3:
                    actual = 'Three Star'
                elif target[i] == 4:
                    actual = 'Four Star'
                else:
                    actual = 'Five Star'
                classifications_file.write('Comment: {}\tPrediction: {}\tActual Rating: {}\n'.format(comment, pred, actual))
        if evaluate and self.numclasses == 2:
            accuracy, precision1, recall1, f1_1, precision2, recall2, f1_2 = self.metrics(target, predictions, counts=False)
            classifications_file.write('\nEvaluation Metrics:\n')
            classifications_file.write('Accuracy: {}%\nClass 1 (Positive) Precision: {}%\n'
                                       'Class 1 (Positive) Recall: {}%\nClass 1 (Positive) F1-Measure: {}%\n'
                                       'Class 2 (Negative) Precision: {}%\nClass 2 (Negative) Recall: {}%\n'
                                       'Class 2 (Negative) F1-Measure: {}%'.format(accuracy * 100, precision1 * 100,
                                                                                   recall1 * 100, f1_1 * 100,
                                                                                   precision2 * 100, recall2 * 100,
                                                                                   f1_2 * 100))
        if evaluate and self.numclasses == 3:
            accuracy, precision1, recall1, f1_1, precision2, recall2, f1_2, precision3, recall3, f1_3 = self.metrics(target, predictions, counts=False)
            classifications_file.write('\nEvaluation Metrics:\n')
            classifications_file.write('Accuracy: {}%\nClass 1 (Positive) Precision: {}%\n'
                                       'Class 1 (Positive) Recall: {}%\nClass 1 (Positive) F1-Measure: {}%\n'
                                       'Class 2 (Negative) Precision: {}%\nClass 2 (Negative) Recall: {}%\n'
                                       'Class 2 (Negative) F1-Measure: {}%\nClass 3 (Neutral) Precision: {}%\n'
                                       'Class 3 (Neutral) Recall: {}%\nClass 3 (Neutral) F1-Measure: {}%\n'.format(accuracy * 100, precision1 * 100,
                                                                                   recall1 * 100, f1_1 * 100,
                                                                                   precision2 * 100, recall2 * 100,
                                                                                   f1_2 * 100, precision3 * 100, recall3 * 100, f1_3 * 100))
        if evaluate and self.numclasses == 5:
            accuracy, precision1, recall1, f1_1, precision2, recall2, f1_2, precision3, recall3, f1_3, precision4, recall4, f1_4, precision5, recall5, f1_5 \
            = self.metrics(target, predictions, counts=False)
            classifications_file.write('\nEvaluation Metrics:\n')
            classifications_file.write('Accuracy: {}%\nOne Star Precision: {}%\n'
                                       'One Star Recall: {}%\nOne Star F1-Measure: {}%\n'
                                       'Two Star Precision: {}%\nTwo Star Recall: {}%\n'
                                       'Two Star F1-Measure: {}%\nThree Star Precision: {}%\n'
                                       'Three Star Recall: {}%\nThree Star F1-Measure: {}%\n'
                                       'Four Star Precision: {}%\nFour Star Recall: {}%\n'
                                       'Four Star F1-Measure: {}%\nFive Star Precision: {}%\n'
                                       'Five Star Recall: {}%\nFive Star F1-Measure: {}%\n'.format(accuracy * 100, precision1 * 100,
                                                                                   recall1 * 100, f1_1 * 100,
                                                                                   precision2 * 100, recall2 * 100,
                                                                                   f1_2 * 100, precision3 * 100, recall3 * 100, f1_3 * 100, precision4 * 100,
                                                                                   recall4 * 100, f1_4 * 100, precision5 * 100, recall5 * 100, f1_5 * 100))

    def save_model(self, output_file):
        """ Saves a trained model to a file
        """

        if self.classifier_type and self.classifier_type != 'nn':
            with open(output_file, 'wb') as pickle_file:
                pickle.dump(self.model, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(self.vectorizer, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

        elif self.classifier_type == 'nn':
            with open("trained_nn_model.json", "w") as json_file:
                json_file.write(self.model.to_json()) # Save mode
            self.model.save_weights("trained_nn_weights.h5") # Save weights
            with open('trained_nn_vec_encoder.pickle', 'wb') as pickle_file:
                pickle.dump(self.vectorizer, pickle_file)
                # pickle.dump(self.encoder, pickle_file)
            tar_file = tarfile.open("trained_nn_model.tar", 'w')
            tar_file.add('trained_nn_model.json')
            tar_file.add('trained_nn_weights.h5')
            tar_file.add('trained_nn_vec_encoder.pickle')
            tar_file.close()

            os.remove('trained_nn_model.json')
            os.remove('trained_nn_weights.h5')
            os.remove('trained_nn_vec_encoder.pickle')

    def load_model(self, model_file=None, tar_file=None, saved_vectorizer=None):
        """ Loads a trained model from a file
        """

        with open(model_file, 'rb') as model_file:
            self.model = pickle.load(model_file)
            self.vectorizer = pickle.load(model_file)

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

    def metrics(self, actual_ratings, predicted_ratings):

        info = {}

        if self.numclasses == 2:

            matrix = confusion_matrix(actual_ratings, predicted_ratings)
            tn, fp, fn, tp = matrix[0][0], matrix[0, 1], matrix[1, 0], matrix[1][1]
            info['tp'], info['tn'], info['fp'], info['fn'] = tp, tn, fp, fn
            info['accuracy'] = (tp + tn) * 1.0 / (tp + tn + fp + fn)
            precision1 = (tp * 1.0) / (tp + fp)
            precision2 = (tn * 1.0) / (tn + fn)
            recall1 = (tp * 1.0) / (tp + fn)
            recall2 = (tn * 1.0) / (tn + fp)
            info['precision1'], info['precision2'], info['recall1'], info['recall2'] = \
                precision1, precision2, recall1, recall2
            info['f1_1'] = 2 * ((precision1 * recall1) / (precision1 + recall1))
            info['f1_2'] = 2 * ((precision2 * recall2) / (precision2 + recall2))

        elif self.numclasses == 3:

            matrix = confusion_matrix(actual_ratings, predicted_ratings)
            tp_pos, tp_neg, tp_neu = matrix[0][0], matrix[1, 1], matrix[2, 2]
            f_ba, f_bc, f_ab = matrix[1, 0], matrix[1, 2], matrix[0, 1]
            f_cb, f_ca, f_ac = matrix[2][1], matrix[2,0], matrix[0, 2]
            info['accuracy'] = ((tp_pos + tp_neg + tp_neu) * 1.0) / \
                               (tp_pos + tp_neg + tp_neu + f_ba + f_bc + f_ab + f_cb + f_ca + f_ac)
            precision1 = (tp_pos * 1.0) / (tp_pos + f_ba + f_ca)
            precision2 = (tp_neg * 1.0) / (tp_neg + f_ab + f_cb)
            precision3 = (tp_neu * 1.0) / (tp_neu + f_bc + f_ac)
            info['precision1'], info['precision2'], info['precision3'] = precision1, precision2, precision3

            recall1 = (tp_pos * 1.0) / (tp_pos + f_ab + f_ac)
            recall2 = (tp_neg * 1.0) / (tp_neg + f_ba + f_bc)
            recall3 = (tp_neu * 1.0) / (tp_neu + f_ca + f_cb)
            info['recall1'], info['recall2'], info['recall3'] = recall1, recall2, recall3

            info['f1_1'] = 2 * ((precision1 * recall1) / (precision1 + recall1))
            info['f1_2'] = 2 * ((precision2 * recall2) / (precision2 + recall2))
            info['f1_3'] = 2 * ((precision3 * recall3) / (precision3 + recall3))

        elif self.numclasses == 5:

            matrix = confusion_matrix(actual_ratings, predicted_ratings)

            tp_one, tp_two, tp_three = matrix[0, 0], matrix[1, 1], matrix[2, 2]
            tp_four, tp_five = matrix[3, 3], matrix[4, 4]
            f_ab, f_ac, f_ad, f_ae = matrix[0, 1], matrix[0, 2], matrix[0, 3], matrix[0, 4]
            f_ba, f_bc, f_bd, f_be = matrix[1, 0], matrix[1, 2], matrix[1, 3], matrix[1, 4]
            f_ca, f_cb, f_cd, f_ce = matrix[2, 0], matrix[2, 1], matrix[2, 3], matrix[2, 4]
            f_da, f_db, f_dc, f_de = matrix[3, 0], matrix[3, 1], matrix[3, 2], matrix[3, 4]
            f_ea, f_eb, f_ec, f_ed = matrix[4, 0], matrix[4, 1], matrix[4, 2], matrix[4, 3]

            info['accuracy'] = ((tp_one + tp_two + tp_three + tp_four + tp_five) * 1.0) / \
                               (tp_one + tp_two + tp_three + tp_four + tp_five + f_ab + f_ac
                                + f_ad + f_ae + f_ba + f_bc + f_bd + f_be + f_ca + f_cb + f_cd
                                + f_ce + f_da + f_db + f_dc + f_de + f_ea + f_eb + f_ec + f_ed)

            precision1 = (tp_one * 1.0) / (tp_one + f_ba + f_ca + f_da + f_ea)
            precision2 = (tp_two * 1.0) / (tp_two + f_ab + f_cb + f_db + f_eb)
            precision3 = (tp_three * 1.0) / (tp_three + f_ac + f_bc + f_dc + f_ec)
            precision4 = (tp_four * 1.0) / (tp_four + f_ad + f_bd + f_cd + f_ed)
            precision5 = (tp_five * 1.0) / (tp_five + f_ae + f_be + f_ce + f_de)
            info['precision1'], info['precision2'], info['precision3'], info['precision4'], info['precision5'] = \
                precision1, precision2, precision3, precision4, precision5

            recall1 = (tp_one * 1.0) / (tp_one + f_ab + f_ac + f_ad + f_ae)
            recall2 = (tp_two * 1.0) / (tp_two + f_ba + f_bc + f_bd + f_be)
            recall3 = (tp_three * 1.0) / (tp_three + f_ca + f_cb + f_cd + f_ce)
            recall4 = (tp_four * 1.0) / (tp_four + f_da + f_db + f_dc + f_de)
            recall5 = (tp_five * 1.0) / (tp_five + f_ea + f_eb + f_ec + f_ed)
            info['recall1'], info['recall2'], info['recall3'], info['recall4'], info['recall5'] = \
                recall1, recall2, recall3, recall4, recall5

            info['f1_1'] = 2 * ((precision1 * recall1) / (precision1 + recall1))

            if precision2 + recall2 == 0:
                info['f1_2'] = 0

            else:
                info['f1_2'] = 2 * ((precision2 * recall2) / (precision2 + recall2))

            info['f1_3'] = 2 * ((precision3 * recall3) / (precision3 + recall3))
            info['f1_4'] = 2 * ((precision4 * recall4) / (precision4 + recall4))
            info['f1_5'] = 2 * ((precision5 * recall5) / (precision5 + recall5))

        return info

