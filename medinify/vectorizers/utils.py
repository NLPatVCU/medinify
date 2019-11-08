
import os
import numpy as np


def find_embeddings():
    """
    Searches of pretrained embeddings file in medinify/data/embeddings folder
    :return: abspath (str) absolute path to embeddings file or None if not found
    """
    abspath = None
    for file in os.walk(os.getcwd()):
        if os.path.isdir(file[0]) and file[0][-24:] == 'medinify/data/embeddings':
            directory_path = file[0]
            embeddings_files = os.listdir(directory_path)
            if not embeddings_files:
                raise FileNotFoundError(
                    'No word embeddings found at data/embeddings.')
            elif len(embeddings_files) > 1:
                print('Multiple embedding files found.\n'
                      'Please specify which file to use (enter file name):')
                while True:
                    for filename in embeddings_files:
                        print('\t%s' % filename)
                    chosen_file = input()
                    if chosen_file in embeddings_files:
                        embeddings_file = chosen_file
                        break
                    else:
                        print('Invalid file entered. '
                              'Please specify which file to use (enter file name):')
            else:
                embeddings_file = embeddings_files[0]
            abspath = directory_path + '/' + embeddings_file
    return abspath


def get_lookup_table(w2v):
    """
    :return lookup_table: (np.array) word embedding lookup table
    """
    index_to_word = w2v.index2word
    lookup_table = np.zeros((len(index_to_word) + 1, w2v.vector_size))
    for i, word in enumerate(index_to_word):
        lookup_table[i + 1] = w2v[word]
    return lookup_table


def get_pos_list():
    """
    Gets POS list as user input
    :return: list of parts of speech to remove
    """
    print('No part(s) for speech specified. Please enter part(s) for speech (separate with commas):')
    pos = input()
    pos_list = [x.strip().upper() for x in pos.split(',')]
    return pos_list
