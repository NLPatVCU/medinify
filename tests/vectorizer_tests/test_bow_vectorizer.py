
import pytest
from medinify.vectorizers import BowVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from medinify.datasets import SentimentDataset
import os
from scipy.sparse import csr_matrix

if os.getcwd()[-16:] == 'vectorizer_tests':
    os.chdir('../..')


def test_initialize_vectorizer():
    vectorizer = BowVectorizer()
    assert vectorizer.nlp
    assert vectorizer.stops
    assert type(vectorizer.vectorizer) == CountVectorizer
    assert BowVectorizer.nickname == 'bow'


def test_get_features_not_trained():
    vectorizer = BowVectorizer()
    dataset = SentimentDataset('citalopram.csv')
    with pytest.raises(AttributeError):
        len(vectorizer.vectorizer.vocabulary_)
    features = vectorizer.get_features(dataset)
    assert type(features[0]) == csr_matrix
    assert len(vectorizer.vectorizer.vocabulary_) > 0


def test_get_feature_trained():
    vectorizer = BowVectorizer()
    dataset1 = SentimentDataset('citalopram.csv')
    vectorizer.get_features(dataset1)
    vocab_length = len(vectorizer.vectorizer.vocabulary_)
    dataset2 = SentimentDataset('actos.csv')
    vectorizer.get_features(dataset2)
    assert len(vectorizer.vectorizer.vocabulary_) == vocab_length


def test_features_label_length():
    vectorizer = BowVectorizer()
    dataset = SentimentDataset('citalopram.csv')
    labels = vectorizer.get_labels(dataset)
    features = vectorizer.get_features(dataset)
    assert labels.shape[0] == features.shape[0]
