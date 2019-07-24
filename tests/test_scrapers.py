"""
Tests for all drug review scrapers
"""

import os
import pytest
from medinify.scrapers import WebMDScraper, DrugRatingzScraper, DrugsScraper, EverydayHealthScraper


# WebMD Scraper Tests
def test_all_data_collected_webmd():
    """
    Tests that the 'data_collected' attribute functions correctly
    when collecting all data
    """
    scraper = WebMDScraper(collect_user_ids=True, collect_urls=True)
    assert 'comment' in scraper.data_collected
    assert 'rating' in scraper.data_collected
    assert 'drug' in scraper.data_collected
    assert 'user id' in scraper.data_collected
    assert 'url' in scraper.data_collected
    assert 'date' in scraper.data_collected


def test_default_data_collected_webmd():
    """
    Tests that the 'data_collected' attribute functions
    correctly with default parameters
    """
    scraper = WebMDScraper()
    assert 'comment' in scraper.data_collected
    assert 'rating' in scraper.data_collected
    assert 'drug' in scraper.data_collected
    assert 'date' in scraper.data_collected
    assert 'user id' not in scraper.data_collected
    assert 'url' not in scraper.data_collected


def test_no_data_collected_webmd():
    """
    Tests that the 'data_collected' attribute functions
    correctly with all data set to false
    """
    scraper = WebMDScraper(collect_ratings=False, collect_dates=False, collect_drugs=False)
    assert 'comment' in scraper.data_collected
    assert 'rating' not in scraper.data_collected
    assert 'drug' not in scraper.data_collected
    assert 'date' not in scraper.data_collected
    assert 'user id' not in scraper.data_collected
    assert 'url' not in scraper.data_collected


def test_dataset_columns_webmd():
    """
    Tests that dataset columns are assigned correctly
    """
    scraper = WebMDScraper(collect_user_ids=True, collect_urls=True)
    columns = list(scraper.dataset.columns)
    assert 'comment' in columns
    assert 'rating' in columns
    assert 'drug' in columns
    assert 'user id' in columns
    assert 'url' in columns
    assert 'date' in columns


def test_wrong_url_webmd():
    """
    Tests scraping page for none-webmd url
    """
    scraper = WebMDScraper()
    with pytest.raises(AssertionError):
        scraper.scrape_page('https://www.drugs.com/comments/aripiprazole/abilify.html?page=1')


def test_no_reviews_webmd():
    """
    Tests scraping page with no reviews
    """
    scraper = WebMDScraper()
    assert scraper.scrape_page('https://www.webmd.com/drugs/drugreview-148631-'
                               'H2Q-oral.aspx?drugid=148631&drugname=H2Q-oral') == 1


def test_scrape_one_page_webmd():
    """
    Tests scraping one page of reviews
    """
    scraper = WebMDScraper()
    scraper.scrape_page(
        'https://www.webmd.com/drugs/drugreview-1701-citalopram-oral.aspx?drugid=1701&drugname=citalopram-oral')
    assert scraper.dataset['comment'].shape[0] == 5
    assert scraper.dataset['rating'].shape[0] == 5
    assert scraper.dataset['date'].shape[0] == 5
    assert scraper.dataset['drug'].shape[0] == 5


def test_scrape_two_pages_webmd():
    """
    Tests that review data is appended correctly to dataset
    """
    scraper = WebMDScraper()
    scraper.scrape_page('https://www.webmd.com/drugs/drugreview-1701-citalopram+oral.'
                        'aspx?drugid=1701&drugname=citalopram+oral&sortby=3')
    scraper.scrape_page('https://www.webmd.com/drugs/drugreview-1701-citalopram+oral.'
                        'aspx?drugid=1701&drugname=citalopram+oral&pageIndex=1&sortby=3&conditionFilter=-1')
    assert scraper.dataset['comment'].shape[0] == 10
    assert scraper.dataset['rating'].shape[0] == 10
    assert scraper.dataset['date'].shape[0] == 10
    assert scraper.dataset['drug'].shape[0] == 10


def test_scrape_page_datatypes_webmd():
    """
    Tests that scraped data has the correct datatypes
    """
    scraper = WebMDScraper(collect_urls=True)
    scraper.scrape_page(
        'https://www.webmd.com/drugs/drugreview-1701-citalopram-oral.aspx?drugid=1701&drugname=citalopram-oral')
    for row in scraper.dataset.itertuples():
        assert type(row.comment) == str
        assert type(row.rating) == dict
        assert type(row.drug) == str
        assert type(row.date) == str
        assert type(row.url) == str


def test_scrape_webmd():
    """
    Tests that scrape function adds to dataset
    """
    scraper = WebMDScraper(collect_urls=True)
    scraper.scrape('https://www.webmd.com/drugs/drugreview-5659-methotrexate-sodium-injection.'
                   'aspx?drugid=5659&drugname=methotrexate-sodium-injection')
    assert scraper.dataset['comment'].shape[0] > 5
    assert scraper.dataset['comment'].shape[0] == \
           scraper.dataset['rating'].shape[0] == \
           scraper.dataset['drug'].shape[0] == \
           scraper.dataset['date'].shape[0]


def test_scrape_datatypes_webmd():
    """
    Tests that scraped data has the correct datatypes
    """
    scraper = WebMDScraper(collect_urls=True)
    scraper.scrape('https://www.webmd.com/drugs/drugreview-5659-methotrexate-sodium-injection.'
                   'aspx?drugid=5659&drugname=methotrexate-sodium-injection')
    for row in scraper.dataset.itertuples():
        assert type(row.comment) == str
        assert type(row.rating) == dict
        assert type(row.drug) == str
        assert type(row.date) == str


def test_get_url_no_reviews_webmd():
    """
    Tests get_url() function for drug name with no review page
    """
    scraper = WebMDScraper()
    url = scraper.get_url('ftyjcfglkjvgkn')
    assert len(url) == 0


def test_get_url_one_page_webmd():
    """
    Tests get_url() function for drug name with one review page
    """
    scraper = WebMDScraper()
    url = scraper.get_url('Infant\'s Ibuprofen')
    assert len(url) == 1


def test_get_url_multiple_pages_webmd():
    """
    Tests get_url() function for drug name with multiple review pages
    """
    scraper = WebMDScraper()
    url = scraper.get_url('aleve')
    assert len(url) > 1


