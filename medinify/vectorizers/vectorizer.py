"""
Vectorizers transform Datasets (contain text and labels) into numerical representations
that can be fed into classification algorithms

While certain algorithms do performs better with and/or require input representations
in a particular format, Vectorizers are designed to be independent of classifier type
"""
import spacy
from abc import ABC, abstractmethod


class Vectorizer(ABC):
    """
    The Vectorizer abstract class is a template
    for the functionality of all Vectorizers
    """
    nickname = None  # how particular Vectorizer will be searched for via keyword arguments

    def __init__(self):
        """
        Standard constructor for all Vectorizers
        :attribute nlp:    spacy model, used for tokenizing
        :attribute stops: stop words to remove
        """
        self.nlp = spacy.load('en_core_web_sm')
        with open('./data/english') as sw:
            self.stops = set(sw.read().splitlines())

    @abstractmethod
    def get_features(self, dataset):
        """
        Transforms text from dataset into numeric representation
        :param dataset: (Dataset) dataset containing data to be Vectorized
        :return: numeric representation of texts (type varies)
        """
        pass

    @staticmethod
    def get_labels(dataset):
        """
        Returns numeric labels
        :param dataset: (Dataset) dataset containing data to be Vectorized
        :return: (pd.Series) numeric representation of labels
        """
        return dataset.data_table['label']

    def tokenize(self, text):
        """
        Lower-cases, removes stopwords, and tokenizes instance of text
        :param text: (str) instance of text from Dataset
        :return: (list) tokens
        """
        tokens = [token.orth_ for token in self.nlp.tokenizer(text.lower())
                  if token.orth_ not in self.stops and not token.is_punct | token.is_space]
        return tokens




