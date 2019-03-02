"""
Tests for the ReviewDataset()
"""

import os
from datetime import date
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
    final_dataset_name = 'doxil-dataset-' + str(date.today()) + '.pickle'
    filenames = [
        'doxil-dataset.pickle', 'doxil-reviews.json', 'doxil-reviews.csv',
        'test-url-dataset.pickle', final_dataset_name
    ]

    for name in filenames:
        if os.path.exists(name):
            os.remove(name)


def test_init_name(dataset):
    """Test the name is lowercased during init"""
    assert dataset.drug_name == 'test'


def test_save(dataset):
    """Test save"""
    dataset.drug_name = "doxil"
    dataset.collect(
        'https://www.webmd.com/drugs/drugreview-12120-Doxil-intravenous.aspx?drugid=12120&drugname=Doxil-intravenous',
        True)
    dataset.save()
    assert os.path.exists('doxil-dataset.pickle')
    assert dataset.meta['drugs'] == ['doxil']
    assert dataset.meta['startTimestamp']
    assert dataset.meta['endTimestamp']
    assert not dataset.meta['locked']


def test_final_save(dataset):
    """Test final save"""
    dataset.drug_name = 'doxil'
    dataset.final_save()
    assert os.path.exists('doxil-dataset-' + str(date.today()) + '.pickle')


def test_load(dataset):
    """Test load"""
    dataset.save()
    dataset.reviews = []
    assert not dataset.reviews

    dataset.load()
    assert dataset.reviews


def test_write_file_json(dataset):
    """Test write json file"""
    dataset.drug_name = 'doxil'
    dataset.write_file('json')
    assert os.path.exists('doxil-reviews.json')


def test_write_file_csv(dataset):
    """Test write csv file"""
    dataset.drug_name = 'doxil'
    dataset.write_file('csv')
    assert os.path.exists('doxil-reviews.csv')


def test_remove_empty_comments(dataset):
    """Test remove empty comments"""
    dataset.remove_empty_comments()
    empty_comments = 0

    for review in dataset.reviews:
        if not review['comment']:
            empty_comments += 1

    assert empty_comments == 0


def test_print_meta(dataset):
    """Test the meta print"""
    dataset.print_meta()

def test_meta(dataset):
    """Test that meta data has been created properly"""
    dataset.load()
    assert dataset.meta['drugs'] == ['test']
    assert dataset.meta['startTimestamp']
    assert dataset.meta['endTimestamp']
    assert not dataset.meta['locked']

def test_lock(dataset):
    """Test that final datasets are properly locked"""
    reviews = dataset.reviews
    dataset.meta['locked'] = True
    dataset.drug_name = "doxil"

    dataset.collect(
        'https://www.webmd.com/drugs/drugreview-12120-Doxil-intravenous.aspx?drugid=12120&drugname=Doxil-intravenous'
    )
    dataset.collect_all_common_reviews()
    dataset.save()
    dataset.final_save()

    assert reviews == dataset.reviews
    assert not os.path.exists('doxil-dataset.pickle')
    assert not os.path.exists('doxil-dataset-' + str(date.today) + '.pickle')


def test_collect_urls(dataset):
    """Test collect urls"""
    dataset.drug_name = 'test-url'
    dataset.collect_urls('test-urls.csv', start=35)
    assert os.path.exists('test-url-dataset.pickle')
