
import spacy
from nltk.corpus import stopwords
from abc import ABC, abstractmethod


class Processor(ABC):

    nickname = None

    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.stops = stopwords.words('english')

    @abstractmethod
    def get_features(self, dataset):
        pass

    def get_labels(self, dataset):
        return dataset.data_table['label']

    def tokenize(self, comment):
        tokens = [token.orth_ for token in self.nlp.tokenizer(comment.lower())
                  if token.orth_ not in self.stops and not token.is_punct | token.is_space]
        return tokens




