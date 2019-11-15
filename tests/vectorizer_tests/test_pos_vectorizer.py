
import pytest
from medinify.vectorizers import PosVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from medinify.datasets import SentimentDataset
import os
from scipy.sparse import csr_matrix

if os.getcwd()[-16:] == 'vectorizer_tests' or os.getcwd()[-13:] == 'scraper_tests':
    os.chdir('../..')


def test_initialize_vectorizer():
    vectorizer = PosVectorizer(pos_list=['NOUN'])
    assert vectorizer.nlp
    assert vectorizer.stops
    assert 'NOUN' in vectorizer.pos_list
    assert type(vectorizer.vectorizer) == CountVectorizer
    assert PosVectorizer.nickname == 'pos'


def test_incorrect_pos():
    with pytest.raises(AssertionError):
        PosVectorizer(pos_list=['NOU'])


def test_get_features():
    vectorizer = PosVectorizer(pos_list=['NOUN'])
    dataset = SentimentDataset('citalopram.csv')
    dataset.data_table = dataset.data_table.iloc[10:20]
    features = vectorizer.get_features(dataset)
    assert type(features[0]) == csr_matrix
    assert len(vectorizer.vectorizer.vocabulary_) > 0

