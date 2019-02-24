"""
Tests for the classifiers
"""

import os
import pytest
from medinify.sentiment import ReviewClassifier

def test_build_dataset_nn():
    """Test build_dataset for neural network"""
    classifier = ReviewClassifier('nn')
    train_data, train_target = classifier.build_dataset('test-reviews.csv')
    assert len(train_data) == len(train_target)
    assert train_data[0][0] in [0, 1]
    assert train_target[0] in [0, 1]

def test_build_dataset_nb():
    """Test build_dataset for naive bayes"""
    classifier = ReviewClassifier('nb')
    dataset = classifier.build_dataset('test-reviews.csv')
    assert list(dataset[0][0].values())[0]
    assert dataset[0][1] == 'pos'
    assert dataset[-1][1] == 'neg'

def test_create_trained_model_nn():
    """Test create trained model for neural network"""
    classifier = ReviewClassifier('nn')
    train_data, train_target = classifier.build_dataset('test-reviews.csv')
    model = classifier.create_trained_model(train_data=train_data, train_target=train_target)
    assert model

def test_create_trained_model_nb():
    """Test create train model for naive bayes"""
    classifier = ReviewClassifier('nb')
    dataset = classifier.build_dataset('test-reviews.csv')
    model = classifier.create_trained_model(dataset=dataset)
    assert model

def test_create_trained_model_rf():
    """Test create trained model for random forest"""
    classifier = ReviewClassifier('rf')
    train_data, train_target = classifier.build_dataset('test-reviews.csv')
    model = classifier.create_trained_model(train_data=train_data, train_target=train_target)
    assert model

def test_train_nn():
    """Test train for neural network"""
    classifier = ReviewClassifier('nn')
    classifier.train('test-reviews.csv')
    assert classifier.model is not None

def test_train_nb():
    """Test train for naive bayes"""
    classifier = ReviewClassifier('nb')
    classifier.train('test-reviews.csv')
    assert classifier.model is not None

def test_train_rf():
    """Test train for neural network"""
    classifier = ReviewClassifier('rf')
    classifier.train('test-reviews.csv')
    assert classifier.model is not None

def test_evaluate_average_accuracy_nn():
    """Test evaluate average accuracy for neural network"""
    classifier = ReviewClassifier('nn')
    average = classifier.evaluate_average_accuracy('test-reviews.csv')
    assert average > 0
    assert average < 100

def test_evaluate_average_accuracy_nb():
    """Test evaluate average accuracy for naive bayes"""
    classifier = ReviewClassifier('nb')
    average = classifier.evaluate_average_accuracy('test-reviews.csv')
    assert average > 0
    assert average < 100

def test_evaluate_average_accuracy_rf():
    """Test evaluate average accuracy for neural network"""
    classifier = ReviewClassifier('rf')
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
