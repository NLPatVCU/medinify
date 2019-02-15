"""
Tests for the classifiers (currently on nn)
"""

import os
import pytest
from medinify.sentiment import NeuralNetReviewClassifier

def test_vectorize():
    """Test vectorize"""
    classifier = NeuralNetReviewClassifier()
    train_data, train_target = classifier.vectorize('test-reviews.csv')
    assert len(train_data) == len(train_target)
    assert train_data[0][0] in [0, 1]
    assert train_target[0] in [0, 1]

def test_create_trained_model():
    """Test create trained model"""
    classifier = NeuralNetReviewClassifier()
    train_data, train_target = classifier.vectorize('test-reviews.csv')
    model = classifier.create_trained_model(train_data, train_target)
    assert model

def test_train():
    """Test train"""
    classifier = NeuralNetReviewClassifier()
    classifier.train('test-reviews.csv')
    assert classifier.model is not None

def test_evaluate_average_accuracy():
    """Test evaluate average accuracy"""
    classifier = NeuralNetReviewClassifier()
    average = classifier.evaluate_average_accuracy('test-reviews.csv')
    assert average > 0
    assert average < 100

def test_save_model():
    classifier = NeuralNetReviewClassifier()
    classifier.train("test-reviews.csv")
    classifier.save_model()
    assert os.path.exists("trained_nn_model.json")

def test_save_model_weights():
    classifier = NeuralNetReviewClassifier()
    classifier.train("test-reviews.csv")
    classifier.save_model_weights()
    assert os.path.exists("trained_nn_weights.h5")

def test_load_model():
    classifier = NeuralNetReviewClassifier()
    classifier.load_nn_model()
    assert classifier.model is not None
