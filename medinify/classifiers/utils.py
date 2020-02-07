
import numpy as np
import os
from medinify import Config
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


def find_model(model):
    """
    Searches models/ directory for specified model file
    :param path: name of saved model file
    :return: abspath - absolute path to file or None if not found
    """
    for directory in os.walk(Config.ROOT_DIR):
        """
        OS walk with iterate through all of the subdirectories in the file path passed to it. In this case its the data
        folder. directory is a tuple with File[0] being the directory path, File[1] is the directory path's 
        subdirectories and File[2] is the files in the current directory path.
        """
        print(directory)
        # In english, if the model file name matches with on of the files the current file directory, it returns the
        # directory path plus the model file name.
        if model in directory[2]:
            return directory[0] + '/' + model
    # If it doesn't find anything return None. (Maybe consider throwing an error here instead of returning None)
    # It would make debugging easier.
    return None
