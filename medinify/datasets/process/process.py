
import numpy as np
import ast
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import spacy
from nltk.corpus import stopwords


class Process:

    data = None
    comments = []
    ratings = []
    w2v = {}
    count_vectors = {}
    tfidf_vectors = {}
    average_embeddings = {}
    remove_stop_words=True

    def __init__(self, data, w2v_file, remove_stop_words=True):
        """
        Initializes an instance of the class Process
        for processing review ratings / comments
        :param data: DataFrame containing collected review data
        :param w2v_file: w2v_file: path to word2vec file
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

        if w2v_file:
            w2v = KeyedVectors.load_word2vec_format(w2v_file)
            wv = dict(zip(list(w2v.vocab.keys()), w2v.vectors))
            self.w2v = wv

        self.nlp = spacy.load("en_core_web_sm")
        self.stops = set(stopwords.words('english'))
        self.remove_stop_words = remove_stop_words

    def count_vectorize(self):
        """
        Produces count vectors and assosiated ratings vector
        """
        vectorizer = CountVectorizer(tokenizer=self.tokenize(remove_stop_words=self.remove_stop_words))
        count_vectors = vectorizer.fit_transform(self.comments)
        self.count_vectors['data'] = count_vectors
        # Add function to process target (ratings)

    def tfidf_vectorize(self):
        """
        Produces tfidf vectors and assosiated ratings vector
        """
        vectorizer = TfidfVectorizer(tokenizer=self.tokenize(remove_stop_words=self.remove_stop_words))
        tfidf_vectors = vectorizer.fit_transform(self.comments)
        self.tfidf_vectors['data'] = tfidf_vectors
        # Add function to process target (ratings)

    def average_embedding_vectorize(self):
        """
        Produces average embeddings from each comment
        """
        new_ratings = []
        average_embeddings = []
        num_discarded = 0

        for i, comment in enumerate(self.comments):
            print(comment)
            token_embeddings = []
            if type(comment) == str:
                tokens = self.tokenize(comment, remove_stop_words=self.remove_stop_words)
            else:
                continue
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
            new_ratings.append(self.ratings[i])

        print('Discarded {} empty comments.'.format(num_discarded))
        self.average_embeddings['data'] = average_embeddings
        # Add function to process target (ratings)

    def tokenize(self, comment, remove_stop_words=True):
        """
        Runs spacy tokenization
        :param comment: comment being tokenized
        :param remove_stop_words: whether or not to remove stop words
        :return: tokens
        """
        if remove_stop_words:
            tokens = [token.text for token in self.nlp.tokenizer(comment)
                      if token.text not in self.stops and not token.is_punct]
        else:
            tokens = [token.text for token in self.nlp.tokenizer(comment) if not token.is_punct]
        return tokens
