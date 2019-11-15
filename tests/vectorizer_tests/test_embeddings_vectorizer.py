
from medinify.vectorizers import EmbeddingsVectorizer
from medinify.datasets import SentimentDataset
from gensim.models import KeyedVectors
import os
import numpy as np

if os.getcwd()[-16:] == 'vectorizer_tests':
    os.chdir('../..')


def test_initialize_vectorizer():
    vectorizer = EmbeddingsVectorizer()
    assert vectorizer.nlp
    assert vectorizer.stops
    assert type(vectorizer.w2v) == KeyedVectors
    assert EmbeddingsVectorizer.nickname == 'embedding'


def test_get_features():
    vectorizer = EmbeddingsVectorizer()
    embedding_dim = vectorizer.w2v.vector_size
    dataset = SentimentDataset('citalopram.csv')
    num_texts = dataset.data_table.shape[0]
    embeddings = vectorizer.get_features(dataset)
    assert embeddings.shape[0] == num_texts
    assert embeddings.shape[1] == embedding_dim
    assert type(embeddings) == np.ndarray


def test_features_and_labels():
    vectorizer = EmbeddingsVectorizer()
    dataset = SentimentDataset('citalopram.csv')
    num_texts = dataset.data_table.shape[0]
    labels = vectorizer.get_labels(dataset)
    features = vectorizer.get_features(dataset)
    assert features.shape[0] == num_texts
    assert labels.shape[0] == num_texts
