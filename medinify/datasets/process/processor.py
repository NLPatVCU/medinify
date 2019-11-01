
import numpy as np
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from abc import ABC, abstractmethod


class Processor(ABC):

    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.stops = stopwords.words('english')

    @abstractmethod
    def get_features(self, dataset):
        pass

    @abstractmethod
    def get_labels(self, dataset):
        pass

    def tokenize(self, comment):
        tokens = [token.orth_ for token in self.nlp.tokenizer(comment.lower())
                  if token.orth_ not in self.stops and not token.is_punct | token.is_space]
        return tokens

    """
    def process_count_vectors(self, comments):
        try:
            self.count_vectorizer.vocabulary_
        except AttributeError:
            self.count_vectorizer.fit(comments)
        count_vectors = self.count_vectorizer.transform(comments)
        return count_vectors

    def get_average_embeddings(self, comments, w2v):
        embeddings = np.zeros((comments.shape[0], 100))
        for i, comment in enumerate(comments):
            tokens = self.tokenize(comment)
            all_embeddings = []
            for token in tokens:
                try:
                    all_embeddings.append(w2v[token])
                except KeyError:
                    continue
            if len(all_embeddings) == 0:
                continue
            else:
                embeddings[i] = np.average(all_embeddings, axis=0)
        return embeddings

    def tokenize(self, comment):
        tokens = [token.orth_ for token in self.nlp.tokenizer(comment.lower())
                  if token.orth_ not in self.stops and not token.is_punct | token.is_space]
        return tokens
    """

    """
    def get_pos_vectors(self, comments, ratings):
        assert config.POS, 'No part of speech specified when constructing Dataset'

        comments = list(comments)
        ratings = np.asarray(ratings)
        target = process_rating(ratings)
        review = namedtuple('review', 'comment, data, target')
        reviews = np.empty(len(comments), dtype=tuple)

        pos_strings = []
        for comment in comments:
            only_pos = ' '.join([token.text for token in self.nlp(comment.lower())
                                 if token.text not in self.stops and not token.is_punct
                                 and token.pos_ == config.POS])
            pos_strings.append(only_pos)

        if not self.pos_vectorizer:
            pos_vectorizer = CountVectorizer(tokenizer=self.tokenize)
            data = np.asarray([x.todense() for x in pos_vectorizer.fit_transform(comments)]).squeeze(1)
            self.pos_vectorizer = pos_vectorizer
        else:
            data = np.asarray([x.todense() for x in self.pos_vectorizer.transform(comments)]).squeeze(1)

        for i in range(reviews.shape[0]):
            datum = review(comment=comments[i], target=target[i], data=data[i])
            reviews[i] = datum

        return reviews
    """


"""
def process_rating(ratings):
    targets = np.empty(ratings.shape[0])
    for i, rating in enumerate(ratings):
        if type(rating) == str:
            target = float(ast.literal_eval(rating)[config.RATING_TYPE])
        else:
            target = rating
        if config.NUM_CLASSES == 2:
            if target >= config.POS_THRESHOLD:
                target = 1
            elif target <= config.NEG_THRESHOLD:
                target = 0
            elif config.NEG_THRESHOLD < target < config.POS_THRESHOLD:
                target = None
        elif config.NUM_CLASSES == 3:
            if target >= config.POS_THRESHOLD:
                target = 2
            elif target <= config.NEG_THRESHOLD:
                target = 0
            elif config.NEG_THRESHOLD < target < config.POS_THRESHOLD:
                target = 1
        elif config.NUM_CLASSES == 5:
            target = target - 1
        targets[i] = target

    return targets
"""


