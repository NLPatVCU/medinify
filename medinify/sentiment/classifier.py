
import numpy as np
import pickle
import os
from medinify.datasets.dataset import Dataset
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from medinify import config


class Classifier:
    """
    The classifier class implements three SkLearn-Based sentiment classifiers
    MultinomialNaiveBayes, Random Forest, and SVC
    For training, evaluation, and validation (k-fold)

    Attributes:
        classifier_type: what type of classifier to train ('nb', 'rf', or 'svm')
        count_vectorizer: turns strings into count vectors
        tfidf_vectorizer: turns strings into tfidf vectors
        pos_count_vectorizer: turns string into count vectors for a certain part of speech
        dataset: the classifier's Dataset (defines for data is loaded and processed)
        pos: part of speech if using pos count vectors
    """

    classifier_type = None
    count_vectorizer = None
    tfidf_vectorizer = None
    pos_count_vectorizer = None

    def __init__(self, classifier_type=None, w2v_file=None, pos=None):
        assert classifier_type in ['nb', 'rf', 'svm'], 'Classifier Type must be \'nb\', \'rf\', or \'svm\''
        self.classifier_type = classifier_type
        self.dataset = Dataset(w2v_file=w2v_file, pos=pos)

    def fit(self, output_file, reviews_file=None, data=None, target=None):
        """
        Trains a model on review data
        :param output_file: path to output trained model
        :param reviews_file: path to csv containing training data
        :param data: data ndarray
        :param target: target ndarray
        """
        if bool(reviews_file):
            data, target = self.load_data(reviews_file)

        model = None
        if self.classifier_type == 'nb':
            model = MultinomialNB()
        elif self.classifier_type == 'rf':
            model = RandomForestClassifier(n_estimators=100)
        elif self.classifier_type == 'svm':
            model = SVC(kernel='rbf', C=10, gamma=0.01)

        print('Fitting model...')
        model.fit(data, target)
        self.save_model(model, output_file)
        print('Model fit.')

    def evaluate(self, trained_model_file, eval_reviews_csv=None, data=None, target=None, verbose=True):
        """
        Evaluates the accuracy, precision, recall, and F-1 score
        of a trained model over a review dataset
        :param trained_model_file: path to file with trained model
        :param eval_reviews_csv: path to csv of review data being evaluated
        :param data: ndarray of data
        :param target: ndarray of target
        :param verbose: whether or not to print metrics
        :return eval_metrics: calculated evaluation metrics (accuracy, precision, recall, f_measure)
        """
        if eval_reviews_csv:
            data, target = self.load_data(eval_reviews_csv)

        trained_model = self.load_model(trained_model_file)
        predictions = trained_model.predict(data)
        accuracy = accuracy_score(target, predictions) * 100
        precisions = {'Class {}'.format(i + 1): score * 100 for i, score in
                      enumerate(precision_score(target, predictions, average=None))}
        recalls = {'Class {}'.format(i + 1): score * 100 for i, score in
                   enumerate(recall_score(target, predictions, average=None))}
        f_scores = {'Class {}'.format(i + 1): score * 100 for i, score in
                    enumerate(f1_score(target, predictions, average=None))}
        matrix = confusion_matrix(target, predictions)

        if verbose:
            print('\nEvaluation Metrics:\n')
            print('Accuracy: {:.4f}%'.format(accuracy))
            print('\n'.join(['{} Precision: {:.4f}%'.format(x[0], x[1]) for x in list(precisions.items())]))
            print('\n'.join(['{} Recall: {:.4f}%'.format(x[0], x[1]) for x in list(recalls.items())]))
            print('\n'.join(['{} F Measure: {:.4f}%'.format(x[0], x[1]) for x in list(f_scores.items())]))
            print('Confusion Matrix:')
            print(matrix)

        return accuracy, precisions, recalls, f_scores, matrix

    def validate(self, review_csv, k_folds=10):
        """
        Runs k-fold cross validation
        :param k_folds: number of folds
        :param review_csv: csv with data to splits, train, and validate on
        """
        data, target = self.load_data(review_csv)
        skf = StratifiedKFold(n_splits=k_folds)

        accuracies = []
        precisions, recalls, f_measures = {}, {}, {}
        if config.NUM_CLASSES == 2:
            precisions = {'Class 1': [], 'Class 2': []}
            recalls = {'Class 1': [], 'Class 2': []}
            f_measures = {'Class 1': [], 'Class 2': []}
        elif config.NUM_CLASSES == 3:
            precisions = {'Class 1': [], 'Class 2': [], 'Class 3': []}
            recalls = {'Class 1': [], 'Class 2': [], 'Class 3': []}
            f_measures = {'Class 1': [], 'Class 2': [], 'Class 3': []}
        elif config.NUM_CLASSES == 5:
            precisions = {'Class 1': [], 'Class 2': [], 'Class 3': [], 'Class 4': [], 'Class 5': []}
            recalls = {'Class 1': [], 'Class 2': [], 'Class 3': [], 'Class 4': [], 'Class 5': []}
            f_measures = {'Class 1': [], 'Class 2': [], 'Class 3': [], 'Class 4': [], 'Class 5': []}
        overall_matrix = None

        num_fold = 1
        for train, test in skf.split(data, target):
            train_data = np.asarray([data[x] for x in train])
            train_target = np.asarray([target[x] for x in train])
            test_data = np.asarray([data[x] for x in test])
            test_target = np.asarray([target[x] for x in test])

            print('Fold {}:'.format(num_fold))
            self.fit('medinify/sentiment/temp_file.txt', data=train_data, target=train_target)
            accuracy, fold_precisions, fold_recalls, fold_f_measures, fold_matrix = self.evaluate(
                'medinify/sentiment/temp_file.txt', data=test_data, target=test_target, verbose=False)

            os.remove('medinify/sentiment/temp_file.txt')

            accuracies.append(accuracy)
            for i in range(config.NUM_CLASSES):
                key_ = 'Class ' + str(i + 1)
                precisions[key_].append(fold_precisions[key_])
                recalls[key_].append(fold_recalls[key_])
                f_measures[key_].append(fold_f_measures[key_])
                if type(overall_matrix) == np.ndarray:
                    overall_matrix += fold_matrix
                else:
                    overall_matrix = fold_matrix

            num_fold += 1

        print_validation_metrics(accuracies, precisions, recalls, f_measures, overall_matrix)

    def classify(self, trained_model_file, reviews_csv, output_file):
        """
        Classifies the sentiment of a reviews csv
        :param trained_model_file: path to file with trained model
        :param reviews_csv: path to reviews being classified
        :param output_file: path to output classifications
        """
        model = self.load_model(trained_model_file)
        data, target, comments = self.load_data(reviews_csv, classifying=True)
        predictions = model.predict(data)

        class_2_sent = {}

        if config.NUM_CLASSES == 2:
            class_2_sent = {0: 'Negative', 1: 'Positive'}
        elif config.NUM_CLASSES == 3:
            class_2_sent = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        elif config.NUM_CLASSES == 5:
            class_2_sent = {0: 'One Star', 1: 'Two Star', 2: 'Three Star', 3: 'Four Star', 4: 'Five Star'}

        with open(output_file, 'w') as f:
            for i, prediction in enumerate(predictions):
                    f.write('Comment: {}\n'.format(comments[i]))
                    f.write('Predicted Class: {}\tActual Class: {}\n\n'.format(
                        class_2_sent[prediction], class_2_sent[target[i]]))

    def save_model(self, trained_model, output_file):
        """
        Saves a trained model and its processor
        :param trained_model: trained model
        :param output_file: path to output saved model file
        """
        with open(output_file, 'wb') as f:
            pickle.dump(trained_model, f, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_model(self, model_file):
        """
        Loads model and processor from pickled format
        :param model_file: path to pickled model file
        :return: loaded model
        """
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
            self.dataset = pickle.load(f)
        return model

    def load_data(self, review_csv, classifying=False):
        """
        Loads and processes data from csv
        :param review_csv: path to csv with review data
        :param classifying: if running classification
        :return: data, target
        """
        self.dataset.load_file(review_csv)

        unprocessed = None
        data, target = None, None
        if config.DATA_REPRESENTATION == 'count':
            data, target, unprocessed = self.dataset.get_count_vectors(classifying=True)
        elif config.DATA_REPRESENTATION == 'tfidf':
            data, target, unprocessed = self.dataset.get_tfidf_vectors(classifying=True)
        elif config.DATA_REPRESENTATION == 'embeddings':
            data, target, unprocessed = self.dataset.get_average_embeddings(classifying=True)
        elif config.DATA_REPRESENTATION == 'pos':
            data, target, unprocessed = self.dataset.get_pos_vectors(classifying=True)

        if classifying:
            return data, target, unprocessed
        else:
            return data, target


def print_validation_metrics(accuracies, precisions, recalls, f_measures, overall_matrix):
    """
    Prints cross validation metrics
    :param accuracies: list of accuracy scores
    :param precisions: list of precision scores per class
    :param recalls: list of recall scores per class
    :param f_measures: list of f-measure scores per class
    :param overall_matrix: total confusion matrix
    """
    print('\n**********************************************************************\n')
    print('Validation Metrics:')
    print('\n\tAverage Accuracy: {:.4f}% +/- {:.4f}%\n'.format(np.mean(accuracies), np.std(accuracies)))
    for i in range(config.NUM_CLASSES):
        key_ = 'Class ' + str(i + 1)
        print('\tClass {} Average Precision: {:.4f}% +/- {:.4f}%'.format(
            i + 1, np.mean(precisions[key_]), np.std(precisions[key_])))
        print('\tClass {} Average Recall: {:.4f}% +/- {:.4f}%'.format(
            i + 1, np.mean(recalls[key_]), np.std(recalls[key_])))
        print('\tClass {} Average F-Measure: {:.4f}% +/- {:.4f}%\n'.format(
            i + 1, np.mean(f_measures[key_]), np.std(f_measures[key_])))
    print('\tOverall Confusion Matrix:\n')
    for row in overall_matrix:
        print('\t{}'.format('\t'.join([str(x) for x in row])))
    print('\n**********************************************************************\n')

