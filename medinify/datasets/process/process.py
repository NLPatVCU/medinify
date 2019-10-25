
import numpy as np
import ast
import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords
from medinify import config
from collections import namedtuple


class Processor:
    # tfidf_vectorizer = None
    # pos_vectorizer = None

    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.stops = stopwords.words('english')
        self.count_vectorizer = CountVectorizer()

    def get_count_vectors(self, comments, ratings):
        comments = list(comments)
        ratings = [ast.literal_eval(rating) for rating in ratings]
        target = process_rating(ratings)
        review = namedtuple('review', 'comment, data, target')
        reviews = np.empty(len(comments), dtype=tuple)

        if not self.count_vectorizer:
            count_vectorizer = CountVectorizer(tokenizer=self.tokenize)
            data = np.asarray([x.todense() for x in count_vectorizer.fit_transform(comments)]).squeeze(1)
            self.count_vectorizer = count_vectorizer
        else:
            data = np.asarray([x.todense() for x in self.count_vectorizer.transform(comments)]).squeeze(1)

        for i in range(reviews.shape[0]):
            datum = review(comment=comments[i], target=target[i], data=data[i])
            reviews[i] = datum

        return reviews

    """
    def get_tfidf_vectors(self, comments, ratings):
        comments = list(comments)
        ratings = np.asarray(ratings)
        target = process_rating(ratings)
        review = namedtuple('review', 'comment, data, target')
        reviews = np.empty(len(comments), dtype=tuple)

        if not self.tfidf_vectorizer:
            tfidf_vectorizer = TfidfVectorizer(tokenizer=self.tokenize)
            data = np.asarray([x.todense() for x in tfidf_vectorizer.fit_transform(comments)]).squeeze(1)
            self.tfidf_vectorizer = tfidf_vectorizer
        else:
            data = np.asarray([x.todense() for x in self.tfidf_vectorizer.transform(comments)]).squeeze(1)

        for i in range(reviews.shape[0]):
            datum = review(comment=comments[i], target=target[i], data=data[i])
            reviews[i] = datum

        return reviews
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

    def get_average_embeddings(self, comments, ratings):
        assert config.WORD_2_VEC, 'No word embeddings file specified when constructing Dataset'

        comments = list(comments)
        ratings = np.asarray(ratings)
        target = process_rating(ratings)
        review = namedtuple('review', 'comment, data, target')
        reviews = np.empty(len(comments), dtype=tuple)

        average_embeddings = []
        for comment in comments:
            tokens = self.tokenize(comment)
            token_embeddings = []
            for token in tokens:
                try:
                    token_embeddings.append(config.WORD_2_VEC[token])
                except KeyError:
                    continue
            if len(token_embeddings) == 0:
                average_embeddings.append(np.zeros(100))
            else:
                average = np.average(token_embeddings, axis=0)
                average_embeddings.append(average)

        data = np.asarray(average_embeddings)

        for i in range(reviews.shape[0]):
            datum = review(comment=comments[i], target=target[i], data=data[i])
            reviews[i] = datum

        return reviews

    def tokenize(self, comment):
        tokens = [token.orth_ for token in self.nlp.tokenizer(comment.lower())
                  if token.orth_ not in self.stops and not token.is_punct | token.is_space]
        return tokens


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


