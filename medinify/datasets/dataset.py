"""
Class for loading, storing, editing, and writing datasets
Works with any csv with specified text and label columns
"""
import pandas as pd
import os
from medinify.datasets.utils import find_csv


class Dataset:
    """
    The Dataset class loads, holds and cleans data from csv files
    An instance of the Dataset class can be passed to a Classifier for training,
    evaluation, validation, and/or classification

    Attributes:
        text_column:    (str) Column name from data csv for text data
        label_column: (str) Column name from data csv for label data
        data_table:         (pandas DataFrame) Where all data is internally stored
    """
    def __init__(self, csv_file=None, text_column='text', label_column='label'):
        """
        Constructor for Dataset
        Sets up what data will be processed as text and label, and loads data
        into DataFrame if csv path is provided
        :param csv_file: (str) Path to csv file with stored data
        :param text_column: (str) name of the csv column containing the text data
        :param label_column: (str) name of the csv column containing the label data
        """
        self.text_column = text_column
        self.label_column = label_column
        if csv_file:
            self.load_file(csv_file)
        else:
            self.data_table = None

    def load_file(self, csv_file):
        """
        Loads a csv file's data into Dataset's internal storage (data_table, pandas DataFrame)
        Removes empty elements with empty text or empty label
        :param csv_file: (str) path to csv file with data
        """
        abspath = find_csv(csv_file)
        if not abspath:
            raise FileNotFoundError('File not found in data/ directory.')
        data_table = pd.read_csv(abspath)
        self.data_table = data_table
        self._clean_data()

    def write_file(self, output_file):
        """
        Writes file of the current internal data (data_table)
        Searches for data/csvs directory, saves csv there
        :param output_file: (str) name to save file to (should end with .csv)
        """
        written = False
        for file in os.walk(os.getcwd()):
            if os.path.isdir(file[0]) and file[0][-18:] == 'medinify/data/csvs':
                directory_path = file[0]
                write_path = directory_path + '/' + output_file
                self.data_table.to_csv(write_path, index=False)
                written = True
        if not written:
            raise NotADirectoryError('data/csvs directory not found.')

    def _remove_empty_elements(self):
        """
        Removes empty elements (text or label) from Dataset
        """
        num_rows = len(self.data_table)
        self.data_table = self.data_table.loc[self.data_table[self.text_column].notnull()]
        self.data_table = self.data_table.loc[self.data_table[self.text_column] != '']
        self.data_table = self.data_table.loc[self.data_table[self.label_column].notnull()]
        num_removed = num_rows - len(self.data_table)
        if num_removed > 0:
            print('Removed %d empty elements(s).' % num_removed)

    def _remove_duplicate_elements(self):
        """
        Removes duplicate texts from Dataset
        """
        num_rows = len(self.data_table)
        self.data_table.drop_duplicates(subset=self.text_column, inplace=True)
        num_removed = num_rows - len(self.data_table)
        if num_removed > 0:
            print('Removed %d duplicate elements(s).' % num_removed)

    def _clean_data(self):
        """
        Removes empty and duplicate elements from Dataset
        :return:
        """
        self._remove_duplicate_elements()
        self._remove_empty_elements()

    def print_stats(self):
        """
        Prints stats about current Dataset, including the number of texts
        and the number of elements per unique label (and percentages)
        """
        labels = self.data_table[self.label_column]
        print('\n******************************************************************************************\n')
        print('Dataset Stats:\n')
        print('Total texts: %d' % len(labels))
        unique_labels = set(labels)
        print('Number of Unique Labels: %d\t(%s)\n' % (
            len(unique_labels), ', '.join([str(x) for x in unique_labels])))

        print('Label Stats:')
        for label in unique_labels:
            num_label = labels.loc[labels == label].shape[0]
            percent = 100 * (num_label / len(labels))
            print('\tLabel: %s\t\tNumber of Instance: %d\t\tPercent of Instances: %.2f%%' % (
                str(label), num_label, percent))
        print('\n******************************************************************************************\n')






