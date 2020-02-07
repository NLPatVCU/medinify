
import os
import numpy as np
from medinify import Config

def find_embeddings(w2v = None):
    """
    Searches of pretrained embeddings file in medinify/data/embeddings folder
    :return: abspath (str) absolute path to embeddings file or None if not found
    """
    if w2v is None:
        w2v = 'w2v.model'
    abspath = None
    for file in os.walk(Config.ROOT_DIR + '/data'):
        file[2]
        if w2v in file[2]:
            return file[0] + '/' + w2v

    return None
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
