
import numpy as np
import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from medinify.datasets.process import Processor


class Model:

    def __init__(self):
        self.model = None
        self.processor = Processor()


class Classifier:

    def __init__(self, classifier_type='nb'):
        assert classifier_type in ['nb', 'rf', 'svm'], 'Classifier Type must be \'nb\', \'rf\', or \'svm\''
        self.classifier_type = classifier_type

    def _get_untrained_model(self):
        model = Model()
        if self.classifier_type == 'nb':
            model.model = MultinomialNB()
        elif self.classifier_type == 'rf':
            model.model = RandomForestClassifier(n_estimators=100, criterion='gini',
                                                 max_depth=None, bootstrap=False, max_features='auto')
        elif self.classifier_type == 'svm':
            model.model = SVC(kernel='rbf', C=10, gamma=0.01)
        return model

    def _get_representations(self, dataset, model):
        features = None
        if dataset.num_classes == 2:
            dataset.dataset = dataset.dataset.loc[dataset.dataset['label'].notnull()]
        if dataset.feature_representation == 'bow':
            features = model.processor.process_count_vectors(dataset.dataset['comment'])
        elif dataset.feature_representation == 'embeddings':
            features = dataset.get_average_embeddings()
        return features

    def fit(self, dataset, output_file=None):
        model = self._get_untrained_model()
        print('Fitting model...')
        if dataset.num_classes == 2:
            dataset.dataset = dataset.dataset.loc[dataset.dataset['label'].notnull()]
        features = self._get_representations(dataset, model)
        model.model.fit(features, dataset.dataset['label'])
        print('Model fit.')
        if output_file:
            self.save_model(model, output_file)
        return model

    def evaluate(self, evaluation_dataset, trained_model=None, trained_model_file=None, verbose=True):
        assert (trained_model or trained_model_file), 'A trained model object or file but be specified'
        if trained_model_file:
            trained_model = self.load_model(trained_model_file)
        features = self._get_representations(evaluation_dataset, trained_model)
        labels = evaluation_dataset.dataset['label']
        accuracy, precisions, recalls, f_scores, matrix = self._evaluate(features, labels, model=trained_model)

        if verbose:
            print('\nEvaluation Metrics:\n')
            print('Accuracy: {:.4f}%'.format(accuracy))
            print('\n'.join(['{} Precision: {:.4f}%'.format(x[0], x[1]) for x in list(precisions.items())]))
            print('\n'.join(['{} Recall: {:.4f}%'.format(x[0], x[1]) for x in list(recalls.items())]))
            print('\n'.join(['{} F Measure: {:.4f}%'.format(x[0], x[1]) for x in list(f_scores.items())]))
            print('Confusion Matrix:')
            print(matrix)

        return accuracy, precisions, recalls, f_scores, matrix

    def _evaluate(self, features, labels, model):
        predictions = model.model.predict(features)
        accuracy = accuracy_score(labels, predictions) * 100
        precisions = {'Class {}'.format(i + 1): score * 100 for i, score in
                      enumerate(precision_score(labels, predictions, average=None))}
        recalls = {'Class {}'.format(i + 1): score * 100 for i, score in
                   enumerate(recall_score(labels, predictions, average=None))}
        f_scores = {'Class {}'.format(i + 1): score * 100 for i, score in
                    enumerate(f1_score(labels, predictions, average=None))}
        matrix = confusion_matrix(labels, predictions)
        return accuracy, precisions, recalls, f_scores, matrix

    def validate(self, dataset, k_folds=10):
        skf = StratifiedKFold(n_splits=k_folds)
        model = self._get_untrained_model()
        if dataset.num_classes == 2:
            dataset.dataset = dataset.dataset.loc[dataset.dataset['label'].notnull()]

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
        for train_indices, test_indices in skf.split(dataset.dataset['comment'], dataset.dataset['label']):
            train_data = dataset.dataset.iloc[train_indices]
            test_data = dataset.dataset.iloc[test_indices]

            fold_accuracy, fold_precisions, fold_recalls, fold_f_scores, fold_matrix = None, None, None, None, None

            print('Fold #%d' % num_fold)
            if dataset.feature_representation == 'bow':
                train_features = model.processor.process_count_vectors(train_data['comment'])
                train_labels = train_data['label']
                model.model.fit(train_features, train_labels)

                test_features = model.processor.process_count_vectors(test_data['comment'])
                test_labels = test_data['label']

                fold_accuracy, fold_precisions, fold_recalls, fold_f_scores, fold_matrix = self._evaluate(
                    test_features, test_labels, model)

            elif dataset.feature_representation == 'embeddings':
                w2v = dataset.w2v
                train_features = model.processor.get_average_embeddings(train_data['comment'], w2v)
                train_labels = train_data['label']
                model.model.fit(train_features, train_labels)

                test_features = model.processor.get_average_embeddings(test_data['comment'], w2v)
                test_labels = test_data['label']

                fold_accuracy, fold_precisions, fold_recalls, fold_f_scores, fold_matrix = self._evaluate(
                    test_features, test_labels, model)

            accuracies.append(fold_accuracy)
            for i in range(dataset.num_classes):
                key_ = 'Class ' + str(i + 1)
                precisions[key_].append(fold_precisions[key_])
                recalls[key_].append(fold_recalls[key_])
                f_measures[key_].append(fold_f_scores[key_])
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
        features = self._get_representations(dataset, trained_model)
        labels = dataset.dataset['label'].to_numpy()
        comments = dataset.dataset['comment']
        predictions = trained_model.model.predict(features)

        with open(output_file, 'w') as f:
            for i in range(labels.shape[0]):
                f.write('Comment: %s\n' % comments.iloc[i])
                f.write('Predicted Class: %d\tActual Class: %d\n\n' % (predictions[i], labels[i]))

    def save_model(self, trained_model, output_file):
        with open(output_file, 'wb') as f:
            pickle.dump(trained_model, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_model(self, model_file):
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

