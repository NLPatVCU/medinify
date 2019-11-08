
import numpy as np
import os


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
