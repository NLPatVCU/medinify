"""
Tests for the ReviewDataset()
"""

import os
import pytest
from medinify.datasets import ReviewDataset

@pytest.fixture()
def dataset():
    """Fixture for standard dataset"""
    print("setup")
    doxil_dataset = ReviewDataset('TEST')
    doxil_dataset.load()

    yield doxil_dataset

    print("teardown")
    filenames = ['doxil-dataset.pickle', 'doxil-reviews.json', 'doxil-reviews.csv']

    for name in filenames:
        if os.path.exists(name):
            os.remove(name)

def test_init_name(dataset):
    """Test the name is lowercased during init"""
    assert dataset.drug_name == 'test'

def test_save(dataset):
    """Test save"""
    dataset.drug_name = "doxil"
    dataset.collect('https://www.webmd.com/drugs/drugreview-12120-Doxil-intravenous.aspx?drugid=12120&drugname=Doxil-intravenous', True)
    dataset.save()
    assert os.path.exists('doxil-dataset.pickle')

def test_load(dataset):
    """Test load"""
    dataset.save()
    dataset.reviews = []
    assert not dataset.reviews

    dataset.load()
    assert dataset.reviews

def test_write_file_json(dataset):
    """Test write json file"""
    dataset.write_file('json')
    
    assert os.path.exists('test-reviews.json')
    
def test_write_file_csv(dataset):
    """Test write csv file"""
    dataset.write_file('csv')
    assert os.path.exists('test-reviews.csv')

def test_remove_empty_comments(dataset):
    """Test remove empty comments"""
    dataset.remove_empty_comments()
    empty_comments = 0

    for review in dataset.reviews:
        if not review['comment']:
            empty_comments += 1

    assert empty_comments == 0

def test_combine_ratings_ttt(dataset):
    """Test combine ratings True True True"""
    review = {'effectiveness': 2, 'ease of use': 3, 'satisfaction': 4}
    dataset.reviews.append(review)
    dataset.combine_ratings()
    assert dataset.reviews[-1]['rating'] == 3

def test_combine_ratings_tft(dataset):
    """Test combine ratings True False True"""
    review = {'effectiveness': 3, 'ease of use': 1, 'satisfaction': 5}
    dataset.reviews.append(review)
    dataset.combine_ratings(True, False, True)
    assert dataset.reviews[-1]['rating'] == 4

def test_balance(dataset):
    """Test balance"""
    dataset.combine_ratings()
    positive_reviews = 0
    negative_reviews = 0

    for review in dataset.reviews:
        if int(review['rating']) > 3:
            positive_reviews += 1
        elif int(review['rating']) < 3:
            negative_reviews += 1

    least_reviews = min([positive_reviews, negative_reviews])

    dataset.balance()

    positive_reviews = 0
    negative_reviews = 0

    for review in dataset.reviews:
        if int(review['rating']) > 3:
            positive_reviews += 1
        elif int(review['rating']) < 3:
            negative_reviews += 1

    assert positive_reviews == least_reviews
    assert negative_reviews == least_reviews