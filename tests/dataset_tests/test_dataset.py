
import pytest
from medinify.datasets import Dataset
from medinify.scrapers import *


def test_init_dataset_default_parameters():
    dataset = Dataset()
    assert type(dataset.scraper) == WebMDScraper
    data_collected = dataset.dataset.columns.values
    assert len(data_collected) == 4
    assert 'comment' in data_collected
    assert 'rating' in data_collected
    assert 'date' in data_collected
    assert 'drug' in data_collected


def test_init_dataset_with_parameters():
    dataset = Dataset(scraper='Drugs', collect_urls=True, collect_user_ids=True)
    assert type(dataset.scraper) == DrugsScraper
    data_collected = dataset.dataset.columns.values
    assert len(data_collected) == 6
    assert 'url' in data_collected
    assert 'user id' in data_collected


def test_error_invalid_parameters_everydayhealth():
    with pytest.raises(AttributeError):
        Dataset(scraper='EverydayHealth', collect_user_ids=True)


def test_error_invalid_parameters_drugratingz():
    with pytest.raises(AttributeError):
        Dataset(scraper='DrugRatingz', collect_user_ids=True)


def test_everydayhealth_scraper():
    dataset = Dataset(scraper='EverydayHealth')
    assert type(dataset.scraper) == EverydayHealthScraper


def test_drugratingz_scraper():
    dataset = Dataset(scraper='DrugRatingz')
    assert type(dataset.scraper) == DrugRatingzScraper


def test_collect_correct_review_data():
    dataset = Dataset()
    dataset.collect('https://www.webmd.com/drugs/drugreview-171648-'
                    'Cabometyx-oral.aspx?drugid=171648&drugname=Cabometyx-oral')
    oldest_review = dataset.dataset.iloc[dataset.dataset.shape[0] - 1]
    assert oldest_review['comment'][:11] == 'I have feet'
    assert oldest_review['rating']['effectiveness'] == 5
    assert oldest_review['rating']['ease of use'] == 2
    assert oldest_review['rating']['satisfaction'] == 5
    assert oldest_review['date'] == '8/1/2017 4:41:12 PM'
    assert oldest_review['drug'] == 'Cabometyx oral'


def test_collect_multiple_drugs():
    dataset = Dataset()
    dataset.collect('https://www.webmd.com/drugs/drugreview-169307-'
                    'Daklinza-oral.aspx?drugid=169307&drugname=Daklinza-oral')
    num_reviews = dataset.dataset.shape[0]
    dataset.collect('https://www.webmd.com/drugs/drugreview-171648-'
                    'Cabometyx-oral.aspx?drugid=171648&drugname=Cabometyx-oral')
    assert dataset.dataset.shape[0] > num_reviews


def test_collect_from_urls():
    pass

