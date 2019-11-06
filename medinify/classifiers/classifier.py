
import numpy as np
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix
from medinify.classifiers import NaiveBayesLearner, RandomForestLearner, SVCLearner
from medinify import process
from medinify.datasets import Dataset
from medinify.classifiers import CNNLearner, ClassificationNetwork


class Model:

    def __init__(self, learner='nb', representation=None):
        self.type = learner
        if learner == 'nb':
            self.learner = NaiveBayesLearner()
        elif learner == 'rf':
            self.learner = RandomForestLearner(
                n_estimators=100, criterion='gini', max_depth=None, bootstrap=False, max_features='auto')
        elif learner == 'svm':
            self.learner = SVCLearner(kernel='rbf', C=10, gamma=0.01)
        elif learner == 'cnn':
            self.learner = CNNLearner()
        else:
            raise AssertionError('model_type must by \'nb\', \'svm\', \'rf\', or \'cnn\'')

        nicknames = [x.nickname for x in process.Processor.__subclasses__()]
        for proc in process.Processor.__subclasses__():
            if representation and proc.nickname == representation:
                self.processor = proc()
            elif proc.nickname == self.learner.default_representation:
                self.processor = proc()
        try:
            self.processor
        except AttributeError:
            raise AttributeError(
                'It looks like you\'re trying to create a Model with an invalid text representation (%s). '
                'Procsessing has been implemented for for the following representation types: %s' %
                (representation, ', '.join(nicknames)))

    def save_model(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.processor, f)
            if self.type == 'cnn':
                pickle.dump(self.learner.network.state_dict(), f)
            else:
                pickle.dump(self.learner, f)

    def load_model(self, path):
        with open(path, 'rb') as f:
            self.processor = pickle.load(f)
            if self.type == 'cnn':
                state_dict = pickle.load(f)
                network = ClassificationNetwork(processor=self.processor)
                network.load_state_dict(state_dict)
                self.learner.network = network
            else:
                self.learner = pickle.load(f)


class Classifier:

    def __init__(self, learner='nb', representation=None):
        assert learner in ['nb', 'rf', 'svm', 'cnn'], \
            'Classifier Type must be \'nb\', \'rf\', \'cnn\', or \'svm\''
        self.learner_type = learner
        self.representation = representation

    def fit(self, dataset, output_file=None):
        model = Model(self.learner_type, self.representation)
        print('Fitting model...')
        features = model.processor.get_features(dataset)
        labels = model.processor.get_labels(dataset)
        model.learner.fit(features, labels, model)
        print('Model fit.')
        if output_file:
            self.save(model, output_file)
        return model

    def evaluate(self, evaluation_dataset, trained_model=None, trained_model_file=None, verbose=True):
        assert (trained_model or trained_model_file), 'A trained model object or file but be specified'
        if trained_model_file:
            trained_model = self.load(trained_model_file)
        features = trained_model.processor.get_features(evaluation_dataset)
        labels = trained_model.processor.get_labels(evaluation_dataset)
        unique_labels = list(set(labels))

        predictions = trained_model.learner.predict(features=features, model=trained_model)
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
        assert (trained_model or trained_model_file), 'A trained model or file but be specified'
        if trained_model_file:
            trained_model = self.load(trained_model_file)
        features = trained_model.processor.get_features(dataset)
        labels = trained_model.processor.get_labels(dataset)
        """
        unique_labels = list(set(labels))
        if self.representation == 'matrix':
            labels, features = trained_model.processor.unpack_samples(features)
        predictions = trained_model.learner.predict(features)
        features = trained_model.processor.get_features(dataset)
        labels = trained_model.processor.get_labels(dataset).to_numpy()
        comments = dataset.data_table[dataset.text_column]
        predictions = trained_model.model.predict(features)

        with open(output_file, 'w') as f:
            for i in range(labels.shape[0]):
                f.write('Comment: %s\n' % comments.iloc[i])
                f.write('Predicted Class: %d\tActual Class: %d\n\n' % (predictions[i], labels[i]))
        """

    @staticmethod
    def save(model, path):
        model.save_model(path)

    def load(self, path):
        model = Model(learner=self.learner_type, representation=self.representation)
        model.load_model(path)
        return model


def print_evaluation_metrics(accuracy, precision_dict, recalls_dict, f_scores_dict, matrix, unique_labels):
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

