
import numpy as np
import ast
import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from gensim.models import KeyedVectors


class Processor:
    """
    For processing review ratings / comments

    Attributes:
        count_vectorizer: trained count vectorizer
        tfidf_vectorizer: trained tfidf vectorizer
        pos_vectorizer: count vectorizer trained over a part-of-speech based vocab
        w2v: dictionary mapping words to vectors for getting average embeddings
        nlp: spacy english model for tokenizing and getting parts of speech
        stops: set of stop words
        num_classes: number or rating classes
        ratings_type: type of rating to process if review date contains multiple types of ratings
    """

    count_vectorizer = None
    tfidf_vectorizer = None
    pos_vectorizer = None
    w2v = None
    nlp = None
    stops = None
    num_classes = None
    rating_type = None
    pos_threshold = None
    neg_threshold = None

    def __init__(self, num_classes=2, rating_type='effectiveness',
                 pos_threshold=4.0, neg_threshold=2.0):
        self.nlp = spacy.load('en_core_web_sm')
        self.stops = stopwords.words('english')
        self.num_classes = num_classes
        self.rating_type = rating_type
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold

    def get_count_vectors(self, comments, ratings):
        """
        Count vectorizes comments
        :param comments: list of comment strings
        :param ratings: list of numeric ratings
        :return: data (ndarray for vectorized comments) and target (ndarray of rating labels)
        """
        comments = list(comments)
        ratings = np.asarray(ratings)

        target, indices = self.process_ratings(ratings)
        comments = [comments[x] for x in indices]

        for i, comment in enumerate(comments):
            if type(comment) == float:
                del comments[i]
                del indices[i]
                del target[i]

        if not self.count_vectorizer:
            count_vectorizer = CountVectorizer(tokenizer=self.tokenize)
            data = np.asarray([x.todense() for x in count_vectorizer.fit_transform(comments)]).squeeze(1)
            self.count_vectorizer = count_vectorizer
        else:
            data = np.asarray([x.todense() for x in self.count_vectorizer.transform(comments)]).squeeze(1)

        target = np.asarray(target)
        return data, target

    def get_tfidf_vectors(self, comments, ratings):
        """
        TF-IDF vectorizes comments
        :param comments: list of comment strings
        :param ratings: list of numeric ratings
        :return: data (ndarray for vectorized comments) and target (ndarray of rating labels)
        """
        comments = list(comments)
        ratings = np.asarray(ratings)

        target, indices = self.process_ratings(ratings)
        comments = [comments[x] for x in indices]

        for i, comment in enumerate(comments):
            if type(comment) == float:
                del comments[i]
                del indices[i]
                del target[i]

        if not self.tfidf_vectorizer:
            tfidf_vectorizer = TfidfVectorizer(tokenizer=self.tokenize)
            data = np.asarray([x.todense() for x in tfidf_vectorizer.fit_transform(comments)]).squeeze(1)
            self.tfidf_vectorizer = tfidf_vectorizer
        else:
            data = np.asarray([x.todense() for x in self.tfidf_vectorizer.transform(comments)]).squeeze(1)

        target = np.asarray(target)
        return data, target

    def get_pos_vectors(self, comments, ratings, part_of_speech):
        """
        Count vectorizes comments using only words of a specific part of speech
        :param comments: list of comment strings
        :param ratings: list of numeric ratings
        :param part_of_speech: sptring representing the part-of-speech being selected
        :return: data (ndarray for vectorized comments) and target (ndarray of rating labels)
        """
        comments = list(comments)
        ratings = np.asarray(ratings)

        target, indices = self.process_ratings(ratings)
        comments = [comments[x] for x in indices]

        for i, comment in enumerate(comments):
            if type(comment) == float:
                del comments[i]
                del indices[i]
                del target[i]

        pos_strings = []
        for comment in comments:
            only_pos = ' '.join([token.text for token in self.nlp(comment.lower())
                                 if token.text not in self.stops and not token.is_punct
                                 and token.pos_ == part_of_speech])
            pos_strings.append(only_pos)

        for i, pos_string in enumerate(pos_strings):
            if pos_string == '':
                del pos_strings[i]
                del target[i]
                del indices[i]

        if not self.pos_vectorizer:
            pos_vectorizer = CountVectorizer(tokenizer=self.tokenize)
            data = np.asarray([x.todense() for x in pos_vectorizer.fit_transform(comments)]).squeeze(1)
            self.pos_vectorizer = pos_vectorizer
        else:
            data = np.asarray([x.todense() for x in self.pos_vectorizer.transform(comments)]).squeeze(1)

        target = np.asarray(target)
        return data, target

    def get_average_embeddings(self, comments, ratings, w2v_file=None):
        """
        Count vectorizes comments using only words of a specific part of speech
        :param comments: list of comment strings
        :param ratings: list of numeric ratings
        :param w2v_file: path to file containing trained word embeddings
        :return: data (ndarray for vectorized comments) and target (ndarray of rating labels)
        """
        assert w2v_file or self.w2v, 'A file containing word vectors must be specified'

        if not self.w2v:
            wv = KeyedVectors.load_word2vec_format(w2v_file)
            w2v = dict(zip(list(wv.vocab.keys()), wv.vectors))
            self.w2v = w2v

        comments = list(comments)
        target, indices = self.process_ratings(ratings)
        comments = [comments[x] for x in indices]

        for i, comment in enumerate(comments):
            if type(comment) == float:
                del comments[i]
                del indices[i]
                del target[i]

        average_embeddings = []
        for comment in comments:
            tokens = self.tokenize(comment)
            token_embeddings = []
            for token in tokens:
                try:
                    token_embeddings.append(self.w2v[token])
                except KeyError:
                    continue
            if len(token_embeddings) == 0:
                average_embeddings.append([])
            else:
                average = np.average(token_embeddings, axis=0)
                average_embeddings.append(average)

        for i, average in enumerate(average_embeddings):
            if type(average) == list:
                del average_embeddings[i]
                del target[i]
                del indices[i]

        data = np.asarray(average_embeddings)
        target = np.asarray(target)
        return data, target

    def process_ratings(self, ratings):
        """
        Processes ratings into label vector
        :param ratings: list of review ratings
        :return: vectorized ratings, indicies indicating where in the original
            list the processed ratings came from
        """
        ratings_and_indices = []
        for i, rating in enumerate(ratings):
            if type(rating) == str:
                ratings_and_indices.append({'target': float(ast.literal_eval(rating)[self.rating_type]), 'index': i})
            else:
                ratings_and_indices.append({'target': rating, 'index': i})

        for i, r_and_i in enumerate(ratings_and_indices):
            if np.isnan(r_and_i['target']):
                del ratings_and_indices[i]

        target = []
        indices = []

        for i, r_and_i in enumerate(ratings_and_indices):
            if self.num_classes == 2:
                if r_and_i['target'] >= self.pos_threshold:
                    target.append('pos')
                    indices.append(r_and_i['index'])
                elif r_and_i['target'] <= self.neg_threshold:
                    target.append('neg')
                    indices.append(r_and_i['index'])
            elif self.num_classes == 3:
                if r_and_i['target'] >= self.pos_threshold:
                    target.append('pos')
                    indices.append(r_and_i['index'])
                elif r_and_i['target'] <= self.neg_threshold:
                    target.append('neg')
                    indices.append(r_and_i['index'])
                elif self.neg_threshold < r_and_i['target'] < self.pos_threshold:
                    target.append('neutral')
                    indices.append(r_and_i['index'])
            elif self.num_classes == 5:
                target.append(str(r_and_i['target']))
                indices.append(r_and_i['index'])

        encoder = LabelEncoder()
        target = list(encoder.fit_transform(target))
        return target, indices

    def tokenize(self, comment):
        """
        Runs spacy tokenization
        :param comment: comment being tokenized
        :return: tokens
        """
        tokens = [token.text for token in self.nlp.tokenizer(comment.lower())
                  if token.text not in self.stops and not token.is_punct]
        return tokens

    """
    def count_vectorize(self, count_vectorizer=None):
        ratings, indices = self.process_ratings()
        if count_vectorizer:
            count_vectors = np.asarray([x.todense() for x in count_vectorizer.transform(
                self.comments)]).squeeze(1)

        else:
            vectorizer = CountVectorizer(tokenizer=self.tokenize)
            count_vectors = np.asarray([[x.todense() for x in vectorizer.fit_transform(
                self.comments)][i] for i in indices]).squeeze(1)
        self.count_vectors['data'] = count_vectors
        self.count_vectors['target'] = ratings

    def tfidf_vectorize(self, tfidf_vectorizer=None):
        ratings, indices = self.process_ratings()
        if tfidf_vectorizer:
            tfidf_vectors = np.asarray([[x.todense() for x in tfidf_vectorizer.transform(
                self.comments)][i] for i in indices]).squeeze(1)
        else:
            vectorizer = TfidfVectorizer(tokenizer=self.tokenize)
            tfidf_vectors = np.asarray([[x.todense() for x in vectorizer.fit_transform(
                self.comments)][i] for i in indices]).squeeze(1)
        self.tfidf_vectors['data'] = tfidf_vectors
        self.tfidf_vectors['target'] = ratings

    def average_embedding_vectorize(self):
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

    def pos_vectorizer(self, pos_vectorizer=None):
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

        if pos_vectorizer:
            vecotrized_comments = np.asarray([x.todense() for x in pos_vectorizer.fit_transform(
                new_comments)]).squeeze(1)
        else:
            vectorizer = CountVectorizer()
            vecotrized_comments = np.asarray([x.todense() for x in vectorizer.fit_transform(
                new_comments)]).squeeze(1)

        self.pos_embeddings['data'] = vecotrized_comments
        self.pos_embeddings['target'] = new_ratings

    def process_ratings(self):
        new_ratings = []
        indices = []

        if type(self.ratings[0]) == dict:
            ratings = [x[self.rating_type] for x in self.ratings]
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
    """
