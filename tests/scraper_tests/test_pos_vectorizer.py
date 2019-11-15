
import pytest
from medinify.vectorizers import PosVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from medinify.datasets import SentimentDataset
import os
from scipy.sparse import csr_matrix

if os.getcwd()[-16:] == 'vectorizer_tests':
    os.chdir('../..')


def test_initialize_vectorizer():
    vectorizer = PosVectorizer()
    assert vectorizer.nlp
    assert vectorizer.stops
    assert type(vectorizer.vectorizer) == CountVectorizer
    assert PosVectorizer.nickname == 'pos'
