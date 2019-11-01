
import numpy as np
import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from medinify.datasets.process import *
from medinify.datasets import Dataset
from medinify.sentiment import CNNClassifier


class Model:

    def __init__(self, model_type='nb', feature_representation='bow'):
        assert model_type in ['nb', 'rf', 'svm', 'cnn'], 'Classifier Type must be \'nb\', \'rf\', \'cnn\', or \'svm\''
        self.model = None
        if model_type == 'nb':
            self.model = MultinomialNB()
        elif model_type == 'rf':
            self.model = RandomForestClassifier(n_estimators=100, criterion='gini',
                                                max_depth=None, bootstrap=False, max_features='auto')
        elif model_type == 'svm':
            self.model = SVC(kernel='rbf', C=10, gamma=0.01)
        elif model_type == 'cnn':
            self.model = CNNClassifier()
            self.network = None
        if feature_representation == 'bow' and model_type != 'cnn':
            self.processor = BowProcessor()
        elif feature_representation == 'embeddings' and model_type != 'cnn':
            self.processor = EmbeddingsProcessor()
        elif model_type == 'cnn':
            self.processor = DataloderProcessor()


class Classifier:

    def __init__(self, classifier_type='nb'):
        assert classifier_type in ['nb', 'rf', 'svm', 'cnn'], 'Classifier Type must be \'nb\', \'rf\', \'cnn\', or \'svm\''
        self.classifier_type = classifier_type

    def fit(self, dataset, output_file=None):
        model = Model(self.classifier_type, dataset.feature_representation)
        print('Fitting model...')
        features = model.processor.get_features(dataset)
        labels = model.processor.get_labels(dataset)

        if self.classifier_type != 'cnn':
            model.model.fit(features, labels)
        else:
            model.network = model.model.fit(features, labels)
        print('Model fit.')
        if output_file:
            self.save_model(trained_model=model, output_file=output_file)
        return model

    def evaluate(self, evaluation_dataset, trained_model=None, trained_model_file=None, verbose=True):
        assert (trained_model or trained_model_file), 'A trained model object or file but be specified'
        if trained_model_file:
            trained_model = self.load_model(model_file=trained_model_file)
        features = trained_model.processor.get_features(evaluation_dataset)
        labels = trained_model.processor.get_labels(evaluation_dataset)
        if self.classifier_type != 'cnn':
            predictions = trained_model.model.predict(features)
            accuracy = accuracy_score(labels, predictions) * 100
            precisions = {'Class {}'.format(i + 1): score * 100 for i, score in
                          enumerate(precision_score(labels, predictions, average=None))}
            recalls = {'Class {}'.format(i + 1): score * 100 for i, score in
                       enumerate(recall_score(labels, predictions, average=None))}
            f_scores = {'Class {}'.format(i + 1): score * 100 for i, score in
                        enumerate(f1_score(labels, predictions, average=None))}
            matrix = confusion_matrix(labels, predictions)

            if verbose:
                print('\nEvaluation Metrics:\n')
                print('Accuracy: {:.4f}%'.format(accuracy))
                print('\n'.join(['{} Precision: {:.4f}%'.format(x[0], x[1]) for x in list(precisions.items())]))
                print('\n'.join(['{} Recall: {:.4f}%'.format(x[0], x[1]) for x in list(recalls.items())]))
                print('\n'.join(['{} F Measure: {:.4f}%'.format(x[0], x[1]) for x in list(f_scores.items())]))
                print('Confusion Matrix:')
                print(matrix)

        else:
            accuracy, precisions, recalls, f_scores, matrix = trained_model.model.evaluate(
                features, trained_model.network)

        return accuracy, precisions, recalls, f_scores, matrix

    def validate(self, dataset, k_folds=10):
        skf = StratifiedKFold(n_splits=k_folds)

        accuracies = []
        precisions, recalls, f_measures = {}, {}, {}
        if dataset.num_classes == 2:
            precisions = {'Class 1': [], 'Class 2': []}
            recalls = {'Class 1': [], 'Class 2': []}
            f_measures = {'Class 1': [], 'Class 2': []}
        elif dataset.num_classes == 3:
            precisions = {'Class 1': [], 'Class 2': [], 'Class 3': []}
            recalls = {'Class 1': [], 'Class 2': [], 'Class 3': []}
            f_measures = {'Class 1': [], 'Class 2': [], 'Class 3': []}
        overall_matrix = None

        num_fold = 1
        for train_indices, test_indices in skf.split(dataset.data_table[dataset.text_column], dataset.data_table[dataset.feature_column]):
            train_dataset = Dataset(num_classes=dataset.num_classes, text_column=dataset.text_column,
                                    feature_column=dataset.feature_column, word_embeddings=dataset.word_embeddings,
                                    feature_representation=dataset.feature_representation)
            train_dataset.data_table = dataset.data_table.iloc[train_indices]
            test_dataset = Dataset(num_classes=dataset.num_classes, text_column=dataset.text_column,
                                   feature_column=dataset.feature_column, word_embeddings=dataset.word_embeddings,
                                   feature_representation=dataset.feature_representation)
            test_dataset.data_table = dataset.data_table.iloc[test_indices]

            print('Fold #%d' % num_fold)
            if self.classifier_type == 'cnn':
                from torchtext.vocab import Vectors
                model = Model(model_type='cnn')
                vectors = Vectors(dataset.word_embeddings)
                model.processor.text_field.build_vocab(train_dataset.data_table['comment'],
                                                       test_dataset.data_table['comment'], vectors=vectors)
            model = self.fit(train_dataset)
            fold_accuracy, fold_precisions, fold_recalls, fold_f_scores, fold_matrix = self.evaluate(
                test_dataset, trained_model=model, verbose=False)

            accuracies.append(fold_accuracy)
            for i in range(dataset.num_classes):
                key_ = 'Class ' + str(i + 1)
                precisions[key_].append(fold_precisions)
                recalls[key_].append(fold_recalls)
                f_measures[key_].append(fold_f_scores)
                if type(overall_matrix) == np.ndarray:
                    overall_matrix += fold_matrix
                else:
                    overall_matrix = fold_matrix

            num_fold += 1

        print_validation_metrics(accuracies, precisions, recalls, f_measures, overall_matrix, dataset.num_classes)

    def classify(self, dataset, output_file, trained_model=None, trained_model_file=None):
        assert (trained_model or trained_model_file), 'A trained model object or file but be specified'
        if trained_model_file:
            trained_model = self.load_model(trained_model_file)
        features = trained_model.processor.get_features(dataset)
        labels = trained_model.processor.get_labels(dataset).to_numpy()
        comments = dataset.data_table[dataset.text_column]
        predictions = trained_model.model.predict(features)

        with open(output_file, 'w') as f:
            for i in range(labels.shape[0]):
                f.write('Comment: %s\n' % comments.iloc[i])
                f.write('Predicted Class: %d\tActual Class: %d\n\n' % (predictions[i], labels[i]))

    @staticmethod
    def save_model(trained_model, output_file):
        with open(output_file, 'wb') as f:
            pickle.dump(trained_model, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_model(model_file):
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        return model


def print_validation_metrics(accuracies, precisions, recalls, f_measures, overall_matrix, num_classes):
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
    for i in range(num_classes):
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

