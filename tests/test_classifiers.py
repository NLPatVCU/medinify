"""
Tests for the classifiers
"""

import os
from medinify.sentiment import ReviewClassifier
import pytest

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
    assert os.path.exists('trained_nb_vectorizer.pickle')
    assert os.path.exists('trained_nb_encoder.pickle')
    os.remove('trained_nb_model.pickle')
    os.remove('trained_nb_vectorizer.pickle')
    os.remove('trained_nb_encoder.pickle')

def test_save_nn_model():
    """Test save nn model"""
    classifier = ReviewClassifier('nn')
    classifier.train('test-reviews.csv')
    classifier.save_model()
    assert os.path.exists('trained_nn_model.json')
    assert os.path.exists('trained_nn_weights.h5')
    assert os.path.exists('trained_nn_vectorizer.pickle')
    assert os.path.exists('trained_nn_encoder.pickle')
    os.remove('trained_nn_model.json')
    os.remove('trained_nn_weights.h5')
    os.remove('trained_nn_vectorizer.pickle')
    os.remove('trained_nn_encoder.pickle')

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

def test_load_model_nb_no_vectorizer():
    """Test load nb model without vectorizer or encoder files"""
    classifier = ReviewClassifier('nb')
    classifier.load_model(pickle_file='test_nb_model.pickle', file_trained_on='test-reviews.csv')
    assert classifier.model

def test_load_model_nn_no_vectorizer():
    """Test load nn model without vectorizer or encoder files"""
    classifier = ReviewClassifier('nn')
    classifier.load_model(json_file='test_nn_model.json',
                          h5_file='test_nn_weights.h5', file_trained_on='test-reviews.csv')
    assert classifier.model

def test_load_model_nb_with_vectorizer():
    """Test load nb model with vectorizer or encoder files"""
    classifier = ReviewClassifier('nb')
    classifier.load_model(pickle_file='test_nb_model.pickle',
                          vectorizer_file='test_nb_vectorizer.pickle',
                          encoder_file='test_nb_encoder.pickle')
    assert classifier.model

def test_load_model_nn_with_vectorizer():
    """Test load nn model with vectorizer or encoder files"""
    classifier = ReviewClassifier('nn')
    classifier.load_model(json_file='test_nn_model.json',
                          h5_file='test_nn_weights.h5',
                          vectorizer_file='test_nn_vectorizer.pickle',
                          encoder_file='test_nn_encoder.pickle')
    assert classifier.model

def test_classify_from_text_file():
    """Test classify comments text file"""
    classifer = ReviewClassifier('nb')
    classifer.load_model(pickle_file='test_nb_model.pickle',
                         vectorizer_file='test_nb_vectorizer.pickle',
                         encoder_file='test_nb_encoder.pickle')
    classifer.classify('classified_comments.txt', comments_text_file='neutral.txt')
    assert os.path.exists('classified_comments.txt')
    os.remove('classified_comments.txt')

def test_classify_nb_from_csv():
    """Test classify using nb model from csv file"""
    classifer = ReviewClassifier('nb')
    classifer.load_model(pickle_file='test_nb_model.pickle',
                         vectorizer_file='test_nb_vectorizer.pickle',
                         encoder_file='test_nb_encoder.pickle')
    classifer.classify('classified_comments.txt', comments_filename='test-reviews.csv')
    assert os.path.exists('classified_comments.txt')
    os.remove('classified_comments.txt')

def test_classify_rf():
    """Test classify using rf"""
    classifer = ReviewClassifier('rf')
    classifer.load_model(pickle_file='test_rf_model.pickle',
                         vectorizer_file='test_rf_vectorizer.pickle',
                         encoder_file='test_rf_encoder.pickle')
    classifer.classify('classified_comments.txt', comments_filename='test-reviews.csv')
    assert os.path.exists('classified_comments.txt')
    os.remove('classified_comments.txt')

def test_classify_nn():
    """Test classify using nn"""
    classifer = ReviewClassifier('nn')
    classifer.load_model(json_file='test_nn_model.json',
                         h5_file='test_nn_weights.h5',
                         vectorizer_file='test_nn_vectorizer.pickle',
                         encoder_file='test_nn_encoder.pickle')
    classifer.classify('classified_comments.txt', comments_filename='test-reviews.csv')
    assert os.path.exists('classified_comments.txt')
    os.remove('classified_comments.txt')

def test_evaluate_accuracy_nb():
    """Test evaluate_accuracy for nb model"""
    classifer = ReviewClassifier('nb')
    classifer.load_model(pickle_file='test_nb_model.pickle',
                         vectorizer_file='test_nb_vectorizer.pickle',
                         encoder_file='test_nb_encoder.pickle')
    score = classifer.evaluate_accuracy('test-reviews.csv')
    assert score > 0
    os.remove('output.log')

def test_evaluate_accuracy_rf():
    """Test evaluate_accuracy for rf model"""
    classifer = ReviewClassifier('rf')
    classifer.load_model(pickle_file='test_rf_model.pickle',
                         vectorizer_file='test_rf_vectorizer.pickle',
                         encoder_file='test_rf_encoder.pickle')
    score = classifer.evaluate_accuracy('test-reviews.csv')
    assert score > 0
    os.remove('output.log')

def test_evaluate_accuracy_nn():
    """Test evaluate_accuracy for nn model"""
    classifer = ReviewClassifier('nn')
    classifer.load_model(json_file='test_nn_model.json',
                         h5_file='test_nn_weights.h5',
                         vectorizer_file='test_nn_vectorizer.pickle',
                         encoder_file='test_nn_encoder.pickle')
    score = classifer.evaluate_accuracy('test-reviews.csv')
    assert score > 0
    os.remove('output.log')
