
import numpy as np
import ast
import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from gensim.models import KeyedVectors


class Process:
    """
    For processing review ratings / comments

    Attributes:
        data: DataFrame containing collected review data
        w2v_file: w2v_file: path to word2vec file
        pos_threshold: positive rating threshold
        neg_threshold: negative rating threshold
        num_classes: number of rating classes
        rating_type: if data has multiple rating types, which one to use
        pos: part of speech to use if using pos arrays
    """

    comments = []
    ratings = []
    data = None
    w2v_file = None
    pos_threshold = None
    neg_threshold = None
    num_class = None
    rating_type = None
    pos = None
    w2v = {}
    count_vectors = {}
    tfidf_vectors = {}
    pos_embeddings = {}
    average_embeddings = {}

    def __init__(self, data, w2v_file, pos_threshold=4.0, neg_threshold=2.0,
                 num_classes=2, rating_type='effectiveness', pos=None,
                 count_vectorize=True, tfidf_vectorize=False, average_emebeddings_vectorize=True,
                 pos_vectorize=False):
        self.data = data
        for i, row in data.iterrows():
            if type(row['rating']) == float:
                if not np.isnan(row['rating']) and type(row['comment']) != float:
                    self.comments.append(row['comment'].lower())
                    self.ratings.append(row['rating'])
            else:
                ratings = ast.literal_eval(row['rating'])
                if type(row['comment']) != float:
                    self.ratings.append(ratings)
                    self.comments.append(row['comment'].lower())

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
        self.pos = pos

        if count_vectorize:
            self.count_vectorize()
            print('Processed count vectors.')
        if tfidf_vectorize:
            self.tfidf_vectorize()
            print('Processed tf-idf vectors')
        if average_emebeddings_vectorize and self.w2v:
            self.average_embedding_vectorize()
            print('Processed average word embeddings.')
        if pos_vectorize and self.pos:
            self.pos_vectorizer()
            print('Processed part-of-speech count vectors.')

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
        self.average_embeddings['data'] = np.asarray(average_embeddings)
        self.average_embeddings['target'] = np.asarray(new_ratings)

    def pos_vectorizer(self):
        """
        Produces pos arrays for comments
        """
        assert self.pos, 'In order to use part of speech arrays, a part of speech must be specified'

        ratings, indices = self.process_ratings()

        pos_comments = []
        for comment in self.comments:
            pos_comment = [token.text for token in self.nlp(comment) if not token.is_punct and
                           token.text not in self.stops and token.pos_ is self.pos]
            pos_comments.append(pos_comment)
        pos_comments = [pos_comments[i] for i in indices]

        new_comments = []
        new_indicies = []

        for i, comment in enumerate(pos_comments):
            if comment:
                new_comments.append(' '.join(comment))
                new_indicies.append(i)
        new_ratings = np.asarray([ratings[i] for i in new_indicies])

        vectorizer = CountVectorizer()
        vecotrized_comments = [x.todense() for x in vectorizer.fit_transform(new_comments)]

        self.pos_embeddings['data'] = vecotrized_comments
        self.pos_embeddings['target'] = new_ratings

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
        encoded = np.asarray(encoder.fit_transform(new_ratings))
        return encoded, indices
