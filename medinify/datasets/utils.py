"""
Medinify dataset utility functions
"""
import os


def find_csv(path):
    """
    Searches data/csvs directory for data file being loaded
    :param path: (str) name of csv file being looked for
    :return absolute_path: (str) (in path found) or None (if not found)
    """
    for file in os.walk(os.getcwd()):
        if os.path.isdir(file[0]) and file[0][-18:] == 'medinify/data/csvs':
            directory_path = file[0]
            absolute_path = directory_path + '/' + path
            if path in os.listdir(directory_path) and os.path.isfile(absolute_path):
                return absolute_path
            else:
                return None
