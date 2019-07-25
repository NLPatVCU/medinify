"""
Tests for all drug review scrapers
"""

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


# Drugs.com Scraper Tests
def test_all_data_collected_drugs():
    """
    Tests data_collected attribute while collecting all review data
    """
    scraper = DrugsScraper(collect_user_ids=True, collect_urls=True)
    assert 'comment' in scraper.data_collected
    assert 'rating' in scraper.data_collected
    assert 'drug' in scraper.data_collected
    assert 'user id' in scraper.data_collected
    assert 'url' in scraper.data_collected
    assert 'date' in scraper.data_collected


def test_default_data_collected_drugs():
    """
    Tests data_collected attribute while collecting default review data
    """
    scraper = DrugsScraper()
    assert 'comment' in scraper.data_collected
    assert 'rating' in scraper.data_collected
    assert 'drug' in scraper.data_collected
    assert 'user id' not in scraper.data_collected
    assert 'url' not in scraper.data_collected
    assert 'date' in scraper.data_collected


def test_no_data_collected_drugs():
    """
    Tests data_collected attribute while all data set to false review data
    """
    scraper = DrugsScraper(collect_ratings=False, collect_dates=False, collect_drugs=False)
    assert 'comment' in scraper.data_collected
    assert 'rating' not in scraper.data_collected
    assert 'drug' not in scraper.data_collected
    assert 'user id' not in scraper.data_collected
    assert 'url' not in scraper.data_collected
    assert 'date' not in scraper.data_collected


def test_scrape_page_incorrect_url_drugs():
    """
    Tests scrape_page() function with incorrect url
    """
    scraper = DrugsScraper()
    with pytest.raises(AssertionError):
        scraper.scrape_page(
            'https://www.webmd.com/drugs/drugreview-1701-citalopram-oral.aspx?drugid=1701&drugname=citalopram-oral')


def test_scrape_page_one_page_drugs():
    """
    Tests scrape_page() function scraping one page for
    correct number of review data
    """
    scraper = DrugsScraper()
    scraper.scrape_page('https://www.drugs.com/comments/naproxen/aleve.html')
    assert scraper.dataset['comment'].shape[0] == 25
    assert scraper.dataset['rating'].shape[0] == 25
    assert scraper.dataset['date'].shape[0] == 25
    assert scraper.dataset['drug'].shape[0] == 25


def test_scrape_page_data_types_drugs():
    """
    Tests scrape_page() function for correct data type collection
    """
    scraper = DrugsScraper(collect_urls=True, collect_user_ids=True)
    scraper.scrape_page('https://www.drugs.com/comments/naproxen/aleve.html')
    for row in scraper.dataset.itertuples():
        assert type(row.comment) == str
        assert type(row.rating) == float or not type(row.rating)
        assert type(row.date) == str
        assert type(row.drug) == str
        assert type(row.url) == str


def test_scrape_page_multiple_pages_drugs():
    """
    Tests scrape_page() function for appending to dataset
    when called multiple times
    """
    scraper = DrugsScraper(collect_urls=True, collect_user_ids=True)
    scraper.scrape_page('https://www.drugs.com/comments/naproxen/aleve.html')
    scraper.scrape_page('https://www.drugs.com/comments/naproxen/aleve.html?page=2')
    assert scraper.dataset['comment'].shape[0] == 50
    assert scraper.dataset['comment'].shape[0] == \
           scraper.dataset['rating'].shape[0] == \
           scraper.dataset['drug'].shape[0] == \
           scraper.dataset['date'].shape[0]


def test_scrape_drugs():
    """
    Tests scrape() function for number of reviews data collected
    """
    scraper = DrugsScraper()
    scraper.scrape('https://www.drugs.com/comments/methotrexate/')
    assert scraper.dataset['comment'].shape[0] > 25
    assert scraper.dataset['comment'].shape[0] == \
           scraper.dataset['rating'].shape[0] == \
           scraper.dataset['drug'].shape[0] == \
           scraper.dataset['date'].shape[0]


def test_get_url_no_url_drugs():
    """
    Tests get_url() function for drug name with no url
    """
    scraper = DrugsScraper()
    url = scraper.get_url('bhkjvghjvg')
    assert len(url) == 0


def test_get_drug_url_drugs():
    """
    Test get_url() function for drug name with a review page
    """
    scraper = DrugsScraper()
    url = scraper.get_url('infants ibuprofen')
    assert len(url) == 1


# DrugRatingz Scraper Tests
def test_user_id_error_drugratingz():
    """
    Tests that setting use_user_ids attribute true causes error
    """
    with pytest.raises(AttributeError):
        DrugRatingzScraper(collect_user_ids=True)


def test_scrape_page_invalid_url_drugratingz():
    """
    Tests for error when scraping invalid url
    """
    scraper = DrugRatingzScraper()
    with pytest.raises(AssertionError):
        scraper.scrape_page('https://www.drugs.com/comments/methotrexate/')


def test_scrape_page_drugratingz():
    """
    Tests that scrape page collects correct number of data
    """
    scraper = DrugRatingzScraper()
    scraper.scrape_page('https://www.drugratingz.com/reviews/17597/Drug-Methotrxate.html')
    assert scraper.dataset['comment'].shape[0] > 0
    assert scraper.dataset['rating'].shape[0] > 0
    assert scraper.dataset['date'].shape[0] > 0
    assert scraper.dataset['drug'].shape[0] > 0


def test_scrape_page_data_types_drugratingz():
    """
    Tests that scrape page function collects correct data types
    """
    scraper = DrugRatingzScraper()
    scraper.scrape_page('https://www.drugratingz.com/reviews/17597/Drug-Methotrxate.html')
    for row in scraper.dataset.itertuples():
        assert type(row.rating) == dict


def test_scrape_multiple_pages_drugratingz():
    """
    Tests that scrape page function appends to dataset correctly
    """
    scraper = DrugRatingzScraper()
    scraper.scrape_page('https://www.drugratingz.com/reviews/17597/Drug-Methotrxate.html')
    scraper.scrape_page('https://www.drugratingz.com/reviews/119/Drug-Prozac.html')
    assert scraper.dataset['comment'].shape[0] > 5


def test_get_url_no_url_drugratingz():
    """
    Tests that get_url() for drug name with no review page
    """
    scraper = DrugRatingzScraper()
    url = scraper.get_url('sdhgfdsasdfgfrd')
    assert len(url) == 0


def test_get_url_one_page_drugratingz():
    """
    Tests that get_url() for drug name with 1 review page
    """
    scraper = DrugRatingzScraper()
    url = scraper.get_url('methotrexate')
    assert len(url) == 1


def test_get_url_multiple_pages_drugratingz():
    """
    Tests that get_url() for drug name with multiple review pages
    """
    scraper = DrugRatingzScraper()
    url = scraper.get_url('prozac')
    assert len(url) > 1


# EverydayHealth Scraper Tests
def test_collect_user_id_error_everydayhealth():
    """
    Tests for error when setting collect_user_id true
    """
    with pytest.raises(AttributeError):
        EverydayHealthScraper(collect_user_ids=True)


def test_scrape_page_everdayhealth():
    """
    Tests that scrape page collects the correct number of data
    """
    scraper = EverydayHealthScraper(collect_urls=True)
    scraper.scrape_page('https://www.everydayhealth.com/drugs/citalopram/reviews')
    assert scraper.dataset['comment'].shape[0] == 20
    assert scraper.dataset['rating'].shape[0] == 20
    assert scraper.dataset['drug'].shape[0] == 20
    assert scraper.dataset['date'].shape[0] == 20
    assert scraper.dataset['url'].shape[0] == 20


def test_scrape_multiple_pages_everydayhealth():
    """
    Tests that scrape_page() appends correctly to dataset
    """
    scraper = EverydayHealthScraper(collect_urls=True)
    scraper.scrape_page('https://www.everydayhealth.com/drugs/citalopram/reviews')
    scraper.scrape_page('https://www.everydayhealth.com/drugs/citalopram/reviews/2')
    assert scraper.dataset['comment'].shape[0] == 40
    assert scraper.dataset['rating'].shape[0] == 40
    assert scraper.dataset['drug'].shape[0] == 40
    assert scraper.dataset['date'].shape[0] == 40
    assert scraper.dataset['url'].shape[0] == 40


def test_get_url_no_page_everydayhealth():
    """
    Tests get_url function for drug name with no review page\
    """
    scraper = EverydayHealthScraper()
    url = scraper.get_url('gkbvfyuilmnb')
    assert len(url) == 0


def test_get_url_everydayhealth():
    """
    Tests that get_url returns the correct url
    """
    scraper = EverydayHealthScraper()
    url = scraper.get_url('methotrexate')
    assert len(url) == 1

