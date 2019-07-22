
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import ast

from gensim.models import KeyedVectors


class Process(ABC):

    data = None
    comments = []
    ratings = []

    def __init__(self, data):
        """
        Initializes an instance of the abstract class Process
        for processing review ratings / comments
        :param data: DataFrame containing collected review data
        """
        self.data = data
        for i, row in data.iterrows():
            if type(row['rating']) == float:
                if not np.isnan(row['rating']):
                    self.comments.append(row['comment'])
                    self.ratings.append(row['rating'])
            else:
                ratings = ast.literal_eval(row['rating'])
                self.ratings.append(ratings)
                self.comments.append(row['comment'])


class ProcessComments(Process):

    w2v = {}

    def __init__(self, data, w2v_file=None):
        """
        Initializes ProcessComment object for running processing on comment data
        :param w2v_file: path to word2vec file
        """
        super(ProcessComments, self).__init__(data)
        if w2v_file:
            w2v = KeyedVectors.load_word2vec_format(w2v_file)
            wv = dict(zip(list(w2v.vocab.keys()), w2v.vectors))
            print(wv)

    def bow_vectorize(self):
        pass


class ProcessRatings(Process):

    def do_something(self):
        return

