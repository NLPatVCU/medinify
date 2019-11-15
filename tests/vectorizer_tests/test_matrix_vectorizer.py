
import pytest
from medinify.vectorizers import MatrixVectorizer
from medinify.datasets import SentimentDataset
from gensim.models import KeyedVectors
import os
import numpy as np

if os.getcwd()[-16:] == 'vectorizer_tests':
    os.chdir('../..')


def test_initialize_vectorizer():
    vectorizer = MatrixVectorizer()
    assert vectorizer.nlp
    assert vectorizer.stops
    assert type(vectorizer.w2v) == KeyedVectors
    assert MatrixVectorizer.nickname == 'matrix'
    assert type(vectorizer.index_to_word) == list
    assert len(vectorizer.index_to_word) > 0


def test_get_features():
    vectorizer = MatrixVectorizer()
    dataset = SentimentDataset('citalopram.csv')
    num_texts = dataset.data_table.shape[0]
    features = vectorizer.get_features(dataset)
    assert type(features.iloc[0]) == np.ndarray
    assert features.iloc[0].dtype == int
    # tests that texts len() < 3 are removed
    assert features.shape[0] <= num_texts


def test_indices_to_tokens():
    indices = [3, 45, 6, 7, 9]
    vectorizer = MatrixVectorizer()
    tokens = vectorizer.indices_to_tokens(indices)
    assert type(tokens[0]) == str
    assert tokens.shape[0] == len(indices)
