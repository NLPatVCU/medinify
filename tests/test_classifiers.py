"""
Tests for the classifiers
"""

import os
from medinify.sentiment import ReviewClassifier
import pytest

def test_create_dataset():
    classifier = ReviewClassifier('nb')
    dataset = classifier.create_dataset('test-reviews.csv')
    assert list(dataset[0][0].values())[0]
    assert dataset[0][1] == 'pos'
    assert dataset[-1][1] == 'neg'

def test_split_data_target():
    classifier = ReviewClassifier('nn')
    dataset = classifier.create_dataset('test-reviews.csv')
    train_data, train_target = classifier.split_data_target(dataset)
    assert len(train_data) == len(train_target)
    assert train_data[0][0] in [0, 1]
    assert train_target[0] in [0, 1]

def test_build_dataset():
    classifier = ReviewClassifier('nn')
    train_data, train_target = classifier.build_dataset('test-reviews.csv')
    assert len(train_data) == len(train_target)
    assert train_data[0][0] in [0, 1]
    assert train_target[0] in [0, 1]

def test_create_trained_model_nn():
    """Test create trained model for neural network"""
    classifier = ReviewClassifier('nn')
    train_data, train_target = classifier.build_dataset('test-reviews.csv')
    model = classifier.create_trained_model(train_data=train_data, train_target=train_target)
    assert model

def test_create_trained_model_nb():
    """Test create train model for naive bayes"""
    classifier = ReviewClassifier('nb')
    dataset = classifier.create_dataset('test-reviews.csv')
    model = classifier.create_trained_model(dataset=dataset)
    assert model

def test_create_trained_model_rf():
    """Test create trained model for random forest"""
    classifier = ReviewClassifier('rf')
    train_data, train_target = classifier.build_dataset('test-reviews.csv')
    model = classifier.create_trained_model(train_data=train_data, train_target=train_target)
    assert model

def test_create_trained_model_svm():
    """Test create trained model for support vector machine"""
    classifier = ReviewClassifier('svm')
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
    """Test train for random forest"""
    classifier = ReviewClassifier('rf')
    classifier.train('test-reviews.csv')
    assert classifier.model is not None

def test_train_svm():
    """Test train for support vector machine"""
    classifier = ReviewClassifier('svm')
    classifier.train('test-reviews.csv')
    assert classifier.model is not None

def test_evaluate_average_accuracy_nn():
    """Test evaluate average accuracy for neural network"""
    classifier = ReviewClassifier('nn')
    average = classifier.evaluate_average_accuracy('test-reviews.csv')
    assert average > 0
    assert average < 100
    os.remove('output.log')

def test_evaluate_average_accuracy_nb():
    """Test evaluate average accuracy for naive bayes"""
    classifier = ReviewClassifier('nb')
    average = classifier.evaluate_average_accuracy('test-reviews.csv')
    assert average > 0
    assert average < 100
    os.remove('output.log')

def test_evaluate_average_accuracy_rf():
    """Test evaluate average accuracy for random forest"""
    classifier = ReviewClassifier('rf')
    average = classifier.evaluate_average_accuracy('test-reviews.csv')
    assert average > 0
    assert average < 100
    os.remove('output.log')

def test_evaluate_average_accuracy_svm():
    """Test evaluate average accuracy for support vector machine"""
    classifier = ReviewClassifier('svm')
    average = classifier.evaluate_average_accuracy('test-reviews.csv')
    assert average > 0
    assert average < 100
    os.remove('output.log')

def test_save_nb_model():
    """Test save nb model"""
    classifier = ReviewClassifier('nb')
    classifier.train('test-reviews.csv')
    classifier.save_model()
    assert os.path.exists('trained_nb_model.pickle')
    os.remove('trained_nb_model.pickle')

def test_save_svm_model():
    """Test save svm model"""
    classifier = ReviewClassifier('svm')
    classifier.train('test-reviews.csv')
    classifier.save_model()
    assert os.path.exists('trained_svm_model.pickle')
    os.remove('trained_svm_model.pickle')

def test_save_nn_model():
    """Test save nn model"""
    classifier = ReviewClassifier('nn')
    classifier.train('test-reviews.csv')
    classifier.save_model()
    assert os.path.exists('trained_nn_model.tar')
    os.remove('trained_nn_model.tar')

def test_load_model_nb_no_file():
    """Test load naive bayes model without  model"""
    classifier = ReviewClassifier('nb')
    with pytest.raises(Exception):
        classifier.load_model()

def test_load_model_nn_no_file():
    """Test load nn model without  model"""
    classifier = ReviewClassifier('nn')
    with pytest.raises(Exception):
        classifier.load_model()

def test_load_nb_model():
    """Test load nb model from pickle file"""
    classifier = ReviewClassifier('nb')
    classifier.load_model(pickle_file='test_nb_model.pickle')
    assert classifier.model

def test_load_nn_model():
    """Test load nn model from tar file"""
    classifier = ReviewClassifier('nn')
    classifier.load_model(tar_file='test_nn_model.tar')
    assert classifier.model
    assert classifier.vectorizer
    assert classifier.encoder

def test_load_svm_model():
    """Test load svm model from pickle file"""
    classifier = ReviewClassifier('svm')
    classifier.load_model(pickle_file='test_svm_model.pickle')
    assert classifier.model
    assert classifier.vectorizer
    assert classifier.encoder

def test_classify_from_text_file():
    """Test classify comments text file"""
    classifer = ReviewClassifier('nb')
    classifer.load_model(pickle_file='test_nb_model.pickle')
    classifer.classify('classified_comments.txt', comments_text_file='neutral.txt')
    assert os.path.exists('classified_comments.txt')
    os.remove('classified_comments.txt')

def test_classify_nb_from_csv():
    """Test classify using nb model from csv file"""
    classifer = ReviewClassifier('nb')
    classifer.load_model(pickle_file='test_nb_model.pickle')
    classifer.classify('classified_comments.txt', comments_filename='test-reviews.csv')
    assert os.path.exists('classified_comments.txt')
    os.remove('classified_comments.txt')

def test_classify_rf():
    """Test classify using rf"""
    classifer = ReviewClassifier('rf')
    classifer.load_model(pickle_file='test_rf_model.pickle')
    classifer.classify('classified_comments.txt', comments_filename='test-reviews.csv')
    assert os.path.exists('classified_comments.txt')
    os.remove('classified_comments.txt')

def test_classify_nn():
    """Test classify using nn"""
    classifer = ReviewClassifier('nn')
    classifer.load_model(tar_file='test_nn_model.tar')
    classifer.classify('classified_comments.txt', comments_filename='test-reviews.csv')
    assert os.path.exists('classified_comments.txt')
    os.remove('classified_comments.txt')

def test_evaluate_accuracy_nb():
    """Test evaluate_accuracy for nb model"""
    classifer = ReviewClassifier('nb')
    classifer.load_model(pickle_file='test_nb_model.pickle')
    score = classifer.evaluate_accuracy('test-reviews.csv')
    assert score > 0
    os.remove('output.log')

def test_evaluate_accuracy_rf():
    """Test evaluate_accuracy for rf model"""
    classifer = ReviewClassifier('rf')
    classifer.load_model(pickle_file='test_rf_model.pickle')
    score = classifer.evaluate_accuracy('test-reviews.csv')
    assert score > 0
    os.remove('output.log')

def test_evaluate_accuracy_nn():
    """Test evaluate_accuracy for nn model"""
    classifer = ReviewClassifier('nn')
    classifer.load_model(tar_file='test_nn_model.tar')
    score = classifer.evaluate_accuracy('test-reviews.csv')
    assert score > 0
    os.remove('output.log')
