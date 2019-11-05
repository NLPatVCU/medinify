from medinify.process import *
import numpy as np
from gensim.models import KeyedVectors
import warnings
import os

warnings.filterwarnings("ignore")


class EmbeddingsProcessor(Processor):

    nickname = 'embedding'

    def __init__(self):
        super().__init__()
        embeddings_file = find_embeddings()
        self.w2v = KeyedVectors.load_word2vec_format(embeddings_file)

    def get_features(self, dataset):
        comments = dataset.data_table[dataset.text_column]
        embeddings = np.zeros((dataset.data_table[dataset.text_column].shape[0], self.w2v.vector_size))
        for i, comment in enumerate(comments):
            tokens = self.tokenize(comment)
            all_embeddings = []
            for token in tokens:
                try:
                    all_embeddings.append(self.w2v[token])
                except KeyError:
                    continue
            if len(all_embeddings) == 0:
                continue
            else:
                embeddings[i] = np.average(all_embeddings, axis=0)
        return embeddings


def find_embeddings():
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
