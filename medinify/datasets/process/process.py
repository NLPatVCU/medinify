
import numpy as np
import ast
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import spacy
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder


class Process:

    data = None
    comments = []
    ratings = []
    w2v = {}
    count_vectors = {}
    tfidf_vectors = {}
    average_embeddings = {}
    pos_threshold = None
    neg_threshold = None
    num_classes = None
    rating_type = None

    def __init__(self, data, w2v_file, pos_threshold=4.0, neg_threshold=2.0,
                 num_classes=2, rating_type='effectiveness'):
        """
        Initializes an instance of the class Process
        for processing review ratings / comments
        :param data: DataFrame containing collected review data
        :param w2v_file: w2v_file: path to word2vec file
        :param pos_threshold: positive rating threshold
        :param neg_threshold: negative rating threshold
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

        for i, comment in enumerate(self.comments):
            if type(comment) == float:
                del self.comments[i]
                del self.ratings[i]

        if w2v_file:
            w2v = KeyedVectors.load_word2vec_format(w2v_file)
            wv = dict(zip(list(w2v.vocab.keys()), w2v.vectors))
            self.w2v = wv

        assert num_classes in [2, 3, 5]
        self.num_classes = num_classes
        self.rating_type = rating_type
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold
        self.nlp = spacy.load("en_core_web_sm")
        self.stops = set(stopwords.words('english'))

    def count_vectorize(self):
        """
        Produces count vectors and assosiated ratings vector
        """
        ratings, indices = self.process_ratings()
        vectorizer = CountVectorizer(tokenizer=self.tokenize)
        count_vectors = [[x.todense() for x in vectorizer.fit_transform(self.comments)][i] for i in indices]
        self.count_vectors['data'] = count_vectors
        self.count_vectors['target'] = ratings

    def tfidf_vectorize(self):
        """
        Produces tfidf vectors and assosiated ratings vector
        """
        ratings, indices = self.process_ratings()
        vectorizer = TfidfVectorizer(tokenizer=self.tokenize)
        tfidf_vectors = [[x.todense() for x in vectorizer.fit_transform(self.comments)][i] for i in indices]
        self.tfidf_vectors['data'] = tfidf_vectors
        self.tfidf_vectors['target'] = ratings

    def average_embedding_vectorize(self):
        """
        Produces average embeddings from each comment
        """
        new_ratings = []
        average_embeddings = []
        num_discarded = 0

        ratings, indices = self.process_ratings()
        comments = [self.comments[i] for i in indices]

        for i, comment in enumerate(comments):
            token_embeddings = []
            tokens = self.tokenize(comment)
            for token in tokens:
                try:
                    token_embeddings.append(self.w2v[token])
                except KeyError:
                    continue
            if len(token_embeddings) == 0:
                num_discarded += 1
                continue
            average_embedding = np.average(token_embeddings, axis=0)
            average_embeddings.append(average_embedding)
            new_ratings.append(ratings[i])

        print('Discarded {} empty comments.'.format(num_discarded))
        self.average_embeddings['data'] = average_embeddings
        self.average_embeddings['target'] = new_ratings

    def tokenize(self, comment):
        """
        Runs spacy tokenization
        :param comment: comment being tokenized
        :return: tokens
        """

        tokens = [token.text for token in self.nlp.tokenizer(comment)
                  if token.text not in self.stops and not token.is_punct]
        return tokens

    def process_ratings(self):
        """
        Numerically encodes ratings
        :return: processed ratings and indices
        """
        new_ratings = []
        indices = []

        if type(self.ratings[0]) == dict:
            ratings = [float(x[self.rating_type]) for x in self.ratings]
        else:
            ratings = self.ratings

        if self.num_classes == 2:
            for i, rating in enumerate(ratings):
                if rating >= self.pos_threshold:
                    new_ratings.append('pos')
                    indices.append(i)
                elif rating <= self.neg_threshold:
                    new_ratings.append('neg')
                    indices.append(i)
        elif self.num_classes == 3:
            for i, rating in enumerate(ratings):
                if rating >= self.pos_threshold:
                    new_ratings.append('pos')
                    indices.append(i)
                elif rating <= self.neg_threshold:
                    new_ratings.append('neg')
                    indices.append(i)
                elif self.neg_threshold < rating < self.pos_threshold:
                    new_ratings.append('neutral')
                    indices.append(i)
        elif self.num_classes == 5:
            if not np.amin(ratings) == 1.0 or not np.amax(ratings) == 5.0:
                raise ValueError('Five classes can only be used with ratings data already '
                                 'split into 5 star classes')
            for i, rating in enumerate(ratings):
                new_ratings.append(str(rating))
                indices.append(i)

        encoder = LabelEncoder()
        encoded = encoder.fit_transform(new_ratings)
        return encoded, indices
