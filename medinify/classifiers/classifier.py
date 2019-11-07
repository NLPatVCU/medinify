
import numpy as np
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from medinify import process
from medinify.datasets import Dataset
from medinify.classifiers import CNNLearner, ClassificationNetwork, get_lookup_table
import os

MultinomialNB.default_processor = 'bow'
SVC.default_processor = 'embedding'
RandomForestClassifier.default_processor = 'bow'
CNNLearner.default_processor = 'matrix'


class Model:
    """
    Model class contains a learner (some classifier, like Naive Bayes, Random Forest,
    etc.) and a processor for transforming text data into numeric data

    These have to be put together into the same object because all data used to
    for fitting, evaluating, and classifying with the learner has to be processed in
    the same way
    """
    def __init__(self, learner='nb', representation=None):
        """
        Constructor for Model
        :param learner: (str) classifier type ('nb' - Naive Bayes, 'rf' - Random Forest,
            'svm' - Support Vector Machine, 'cnn' - Convolutional Neural Network)
        :param representation: How text data will be processed ('bow' -
            bag of words, 'embedding' - average embedding, 'matrix' - embedding matrix)
        """
        self.type = learner
        if learner == 'nb':
            self.learner = MultinomialNB()
        elif learner == 'rf':
            self.learner = RandomForestClassifier(
                n_estimators=100, criterion='gini', max_depth=None, bootstrap=False, max_features='auto')
        elif learner == 'svm':
            self.learner = SVC(kernel='rbf', C=10, gamma=0.01)
        elif learner == 'cnn':
            self.learner = CNNLearner()
        else:
            raise AssertionError('model_type must by \'nb\', \'svm\', \'rf\', or \'cnn\'')

        for proc in process.Processor.__subclasses__():
            if representation and proc.nickname == representation:
                self.processor = proc()
            elif proc.nickname == self.learner.default_processor:
                self.processor = proc()
        try:
            self.processor
        except AttributeError:
            print('Invalid feature representation')

    def save_model(self, path):
        """
        Saves trained model to a file
        :param path: (str) file name to save at (all models will be saved to models/ directory)
        """
        with open(path, 'wb') as f:
            pickle.dump(self.processor, f)
            if self.type == 'cnn':
                pickle.dump(self.learner.network.state_dict(), f)
            else:
                pickle.dump(self.learner, f)

    def load_model(self, path):
        """
        Load trained model from file
        :param path: (str) model file name (will search through models/ directory for this file)
        """
        with open(path, 'rb') as f:
            self.processor = pickle.load(f)
            if self.type == 'cnn':
                state_dict = pickle.load(f)
                lookup_table = get_lookup_table()
                network = ClassificationNetwork(lookup_table)
                network.load_state_dict(state_dict)
                self.learner.network = network
            else:
                self.learner = pickle.load(f)


class Classifier:
    """
    Classifier is used to train, evaluate, and validate classification models
    and use trained models for classification
    """
    def __init__(self, learner='nb', representation=None):
        """
        Constructs Classifier
        :param learner: (str) classifier type ('nb' - Naive Bayes, 'rf' - Random Forest,
            'svm' - Support Vector Machine, 'cnn' - Convolutional Neural Network)
        :param representation: How text data will be processed ('bow' -
            bag of words, 'embedding' - average embedding, 'matrix' - embedding matrix)
        """
        assert learner in ['nb', 'rf', 'svm', 'cnn'], \
            'Classifier Type must be \'nb\', \'rf\', \'cnn\', or \'svm\''
        self.learner_type = learner
        self.representation = representation

    def fit(self, dataset, output_file=None):
        """
        Fits a model for features and labels
        :param dataset: (Dataset) dataset containing text and labels to fit model to
        :param output_file: (str) where to save trained model
        """
        model = Model(self.learner_type, self.representation)
        print('Fitting model...')
        features = model.processor.get_features(dataset)
        labels = model.processor.get_labels(dataset)
        model.learner.fit(features, labels)
        print('Model fit.')
        if output_file:
            self.save(model, output_file)
        return model

    def evaluate(self, evaluation_dataset, trained_model=None, trained_model_file=None, verbose=True):
        """
        Evaluates the effectiveness of trained model for classifying a Dataset
        :param evaluation_dataset: (Dataset) data being evaluated over
        :param trained_model: (Model) trained Model
        :param trained_model_file: (str) path to saved model file
        :param verbose: (boolean) whether or not to print results
        :return: accuracy, precision_dict, recalls_dict, f_scores_dict, matrix
        """
        assert (trained_model or trained_model_file), 'A trained model object or file but be specified'
        if trained_model_file:
            trained_model = self.load(trained_model_file)
        features = trained_model.processor.get_features(evaluation_dataset)
        labels = trained_model.processor.get_labels(evaluation_dataset)
        unique_labels = list(set(labels))

        if not self.learner_type == 'cnn':
            predictions = trained_model.learner.predict(features)
        else:
            predictions = trained_model.learner.predict(features, trained_model)
        accuracy = accuracy_score(labels, predictions)
        precisions = precision_score(labels, predictions, average=None, labels=unique_labels)
        precision_dict = dict(zip(unique_labels, precisions))
        recalls = recall_score(labels, predictions, average=None, labels=unique_labels)
        recalls_dict = dict(zip(unique_labels, recalls))
        f_scores = f1_score(labels, predictions, average=None, labels=unique_labels)
        f_scores_dict = dict(zip(unique_labels, f_scores))
        matrix = confusion_matrix(labels, predictions, labels=unique_labels)

        if verbose:
            print_evaluation_metrics(
                accuracy, precision_dict, recalls_dict, f_scores_dict, matrix, unique_labels)

        return accuracy, precision_dict, recalls_dict, f_scores_dict, matrix

    def validate(self, dataset, k_folds=10):
        """
        Runs K-Fold cross validation on a particular dataset
        :param dataset: (Dataset) data to run K-Fold cross validation on
        :param k_folds: (int) number of k-folds
        """
        skf = StratifiedKFold(n_splits=k_folds)
        accuracies = []
        precisions = []
        recalls = []
        f_scores = []
        total_matrix = None

        train_dataset = Dataset(text_column=dataset.text_column, label_column=dataset.label_column)
        test_dataset = Dataset(text_column=dataset.text_column, label_column=dataset.label_column)

        num_fold = 1
        for train_indices, test_indices in skf.split(dataset.data_table[dataset.text_column], dataset.data_table[dataset.label_column]):
            print('\nFold %s:' % num_fold)
            train_data = dataset.data_table.iloc[train_indices]
            train_dataset.data_table = train_data
            test_data = dataset.data_table.iloc[test_indices]
            test_dataset.data_table = test_data
            model = self.fit(train_dataset)
            fold_accuracy, fold_precisions, fold_recalls, fold_f_scores, fold_matrix = self.evaluate(
                test_dataset, trained_model=model, verbose=False)
            accuracies.append(fold_accuracy)
            precisions.append(fold_precisions)
            recalls.append(fold_recalls)
            f_scores.append(fold_f_scores)
            if type(total_matrix) == np.ndarray:
                total_matrix += fold_matrix
            else:
                total_matrix = fold_matrix
            num_fold += 1

        unique_labels = list(precisions[0].keys())
        print_validation_metrics(accuracies, precisions, recalls, f_scores, total_matrix, unique_labels)

    def classify(self, dataset, output_file, trained_model=None, trained_model_file=None):
        """
        Uses trained Model to classify a dataset
        :param dataset: (Dataset) data to classify
        :param output_file: (str) path to write classifications
        :param trained_model: (Model) trained Model
        :param trained_model_file: (str) path to saved model file
        """
        assert (trained_model or trained_model_file), 'A trained model or file but be specified'
        if trained_model_file:
            trained_model = self.load(trained_model_file)
        features = trained_model.processor.get_features(dataset)
        labels = trained_model.processor.get_labels(dataset).to_numpy()
        comments = dataset.data_table[dataset.text_column]
        predictions = trained_model.learner.predict(features, trained_model)

        with open(output_file, 'w') as f:
            for i in range(labels.shape[0]):
                f.write('Comment: %s\n' % comments.iloc[i])
                f.write('Predicted Class: %d\tActual Class: %d\n\n' % (predictions[i], labels[i]))

    @staticmethod
    def save(model, path):
        """
        Save trained model
        :param model: (Model) model to save
        :param path: (str) path to save model
        """
        written = False
        for file in os.walk(os.getcwd()):
            if os.path.isdir(file[0]) and file[0][-15:] == 'medinify/models':
                directory_path = file[0]
                write_path = directory_path + '/' + path
                model.save_model(write_path)
                written = True
        if not written:
            raise NotADirectoryError('models/ directory not found.')

    def load(self, path):
        """
        Load trained model file
        :param path: (str) path to trained model file
        :return model: (Model) loaded model
        """
        model = Model(learner=self.learner_type, representation=self.representation)
        abspath = find_model(path)
        if not abspath:
            raise NotADirectoryError('models/ directory not found.')
        model.load_model(abspath)
        return model


def find_model(path):
    """
    Searches models/ directory for specified model file
    :param path: name of saved model file
    :return: abspath - absolute path to file or None if not found
    """
    for file in os.walk(os.getcwd()):
        if os.path.isdir(file[0]) and file[0][-15:] == 'medinify/models':
            directory_path = file[0]
            absolute_path = directory_path + '/' + path
            if path in os.listdir(directory_path) and os.path.isfile(absolute_path):
                return absolute_path
            else:
                return None


def print_evaluation_metrics(accuracy, precision_dict, recalls_dict, f_scores_dict, matrix, unique_labels):
    """
    Prints evaluation metrics
    """
    print('\n***************************************************\n')
    print('Evaluation Metrics:\n')
    print('\tOverall Accuracy:\t%.2f%%\n' % (accuracy * 100))
    for label in unique_labels:
        print('\t%s label precision:\t%.2f%%' % (str(label), precision_dict[label] * 100))
        print('\t%s label recall:\t%.2f%%' % (str(label), recalls_dict[label] * 100))
        print('\t%s label f-score:\t%.2f%%\n' % (str(label), f_scores_dict[label] * 100))

    print('\tConfusion Matrix:\n')
    for row in matrix:
        print('\t{}'.format('\t'.join([str(x) for x in row])))
    print('\n***************************************************\n')


def print_validation_metrics(accuracies, precisions, recalls, f_scores, total_matrix, unique_labels):
    """
    Prints validation metrics
    """
    print('\n**********************************************************************\n')
    print('Validation Metrics:')
    print('\n\tAverage Accuracy:\t%.4f%% +/- %.4f%%\n' % (np.mean(accuracies) * 100, np.std(accuracies) * 100))
    for label in unique_labels:
        label_precisions = [x[label] for x in precisions]
        label_recalls = [x[label] for x in recalls]
        label_f_scores = [x[label] for x in f_scores]
        print('\n\t%s Label Average Precision:\t%.4f%% +/- %.4f%%' % (
            str(label), np.mean(label_precisions) * 100, np.std(label_precisions) * 100))
        print('\t%s Label Average Recall:\t%.4f%% +/- %.4f%%' % (
            str(label), np.mean(label_recalls) * 100, np.std(label_recalls) * 100))
        print('\t%s Label Average F-Score:\t%.4f%% +/- %.4f%%' % (
            str(label), np.mean(label_f_scores) * 100, np.std(label_f_scores) * 100))
    print('\n\tConfusion Matrix:\n')
    for row in total_matrix:
        print('\t{}'.format('\t'.join([str(x) for x in row])))
    print('\n**********************************************************************\n')

