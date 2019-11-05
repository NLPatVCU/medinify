
import pandas as pd
import os


class Dataset:

    def __init__(self, csv_file=None, text_column='text', label_column='label',  **kwargs):
        self.text_column = text_column
        self.label_column = label_column
        self.args = kwargs
        if csv_file:
            self.load_file(csv_file)
        else:
            self.data_table = None

    def load_file(self, csv_file):
        abspath = find_csv(csv_file)
        if not abspath:
            raise FileNotFoundError('File not found in data/ directory.')
        data_table = pd.read_csv('./data/' + csv_file)
        self.data_table = data_table
        self._clean_data()

    def write_file(self, output_file):
        self.data_table.to_csv('./data/' + output_file, index=False)

    def _remove_empty_elements(self):
        num_rows = len(self.data_table)
        self.data_table = self.data_table.loc[self.data_table[self.text_column].notnull()]
        self.data_table = self.data_table.loc[self.data_table[self.text_column] != '']
        self.data_table = self.data_table.loc[self.data_table[self.label_column].notnull()]
        num_removed = num_rows - len(self.data_table)
        if num_removed > 0:
            print('Removed %d empty elements(s).' % num_removed)

    def _remove_duplicate_elements(self):
        num_rows = len(self.data_table)
        self.data_table.drop_duplicates(subset=self.text_column, inplace=True)
        num_removed = num_rows - len(self.data_table)
        if num_removed > 0:
            print('Removed %d duplicate elements(s).' % num_removed)

    def _clean_data(self):
        self._remove_duplicate_elements()
        self._remove_empty_elements()

    def print_stats(self):
        labels = self.data_table[self.label_column]
        print('\n******************************************************************************************\n')
        print('Dataset Stats:\n')
        print('Total reviews: %d' % len(labels))
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


def find_csv(path):
    for file in os.walk(os.getcwd()):
        if os.path.isdir(file[0]) and file[0][-13:] == 'medinify/data':
            directory_path = file[0]
            absolute_path = directory_path + '/' + path
            if path in os.listdir(directory_path) and os.path.isfile(absolute_path):
                return absolute_path
            else:
                return None





