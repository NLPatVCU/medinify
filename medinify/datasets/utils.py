"""
Medinify dataset utility functions
"""
import os
from medinify import Config


def find_csv(csv):
    """
    Searches data/csvs directory for data file being loaded
    :param path: (str) name of csv file being looked for
    :return absolute_path: (str) (in path found) or None (if not found)
    """
    for directory in os.walk(Config.ROOT_DIR + '/data'):
        """
        OS walk with iterate through all of the subdirectories in the file path passed to it. In this case its the data
        folder. directory is a tuple with File[0] being the directory path, File[1] is the directory path's 
        subdirectories and File[2] is the files in the current directory path.
        """
        print(directory)

        # In english, if the csv file name matches with on of the files the current file directory, it returns the
        # directory path plus the csv file name.
        if csv in directory[2]:
            print(directory[0] + '/' + csv)
            return directory[0] + '/' + csv
    # If it doesn't find anything return None. (Maybe consider throwing an error here instead of returning None)
    # It would make debugging easier.
    return None
