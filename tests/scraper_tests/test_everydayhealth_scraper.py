

import pytest
from medinify.scrapers import EverydayHealthScraper


def test_default_initialization():
    """
    Tests that when the new scraper object is initialized, the correct
    attributes are set (empty list[dict] 'reviews') and the attribute
    collect_urls (boolean) is false by default
    """
    scraper = EverydayHealthScraper()
    assert len(scraper.reviews) == 0
    assert not scraper.collect_urls


def test_initialization_parameters():
    """
    Tests that when a new EverydayHealth scraper object is initialized with a
    non default argument the correct attributes are set (empty list[dict] 'reviews')
    and that the collect_urls attribute (boolean) is set to true
    """
    scraper = EverydayHealthScraper(collect_urls=True)
    assert scraper.collect_urls


def test_incorrect_url():
    """
    Tests that the scrape_page function raises and assertion error when a url is
    entered with an incorrect domain name (not 'https://everydayhealth.com...')
    """
    scraper = EverydayHealthScraper()
    with pytest.raises(AssertionError):
        scraper.scrape_page('https://www.drugs.com/comments/aripiprazole/abilify.html?page=1')


def test_no_reviews():
    """
    Tests that the scrape page function returns 'None' when no reviews are
    found on the page
    """
    scraper = EverydayHealthScraper()
    returned = scraper.scrape_page('https://www.everydayhealth.com/drugs/triprolidine/reviews')
    assert not returned


def test_scrape_correct_review_data():
    """
    Tests to make sure that the last review in the scraped reviews list has
    the correct data when the scrape function is called
    (this data is from the oldest review of the drug)
    """
    scraper = EverydayHealthScraper(collect_urls=True)
    scraper.scrape('https://www.everydayhealth.com/drugs/ciclopirox-topical/reviews')
    assert scraper.reviews[-1]['comment'][:10] == 'After OVER'
    assert scraper.reviews[-1]['comment'][-10:] == 'inally hav'
    assert scraper.reviews[-1]['rating'] == 5
    assert scraper.reviews[-1]['date'] == '5/22/2015 4:18:19 AM'


def test_scrape_page_default_parameters():
    """
    Tests to make sure that calling the scrape_page function on a scraper object
    with a default parameter collects the correct types of data ('comment', 'rating',
    'date', and 'drug') and that the correct number of reviews were collected (20)
    """
    scraper = EverydayHealthScraper()
    scraper.scrape_page('https://www.everydayhealth.com/drugs/allegra/reviews')
    data_collected = list(scraper.reviews[0].keys())
    assert len(data_collected) == 4
    assert 'comment' in data_collected
    assert 'rating' in data_collected
    assert 'date' in data_collected
    assert 'drug' in data_collected
    assert len(scraper.reviews) == 20


def test_scrape_page_with_parameters():
    """
    Tests to make sure that calling the scrape_page function on a scraper object with a
    non-default parameter (collect_url true) collects the correct types of data
    ('comment', 'rating, 'date', 'drug', and 'url')
    """
    scraper = EverydayHealthScraper(collect_urls=True)
    scraper.scrape_page('https://www.everydayhealth.com/drugs/hydroxyzine/reviews')
    data_collected = list(scraper.reviews[0].keys())
    assert len(data_collected) == 5
    assert 'url' in data_collected


def test_scrape_empty_reviews():
    """
    Tests to make sure that the scrape function would discard the reviews
    of a scraper object that already has data collected in 'reviews'
    """
    scraper = EverydayHealthScraper()
    scraper.scrape('https://www.everydayhealth.com/drugs/phenadoz/reviews')
    num_reviews = len(scraper.reviews)
    scraper.scrape('https://www.everydayhealth.com/drugs/phenadoz/reviews')
    assert num_reviews == len(scraper.reviews)


def test_scrape_invalid_url_no_title():
    """
    Tests that when the scrape function is called on a url that lacks a title
    (invalid url), it raises an AttributeError and returns 0
    """
    scraper = EverydayHealthScraper()
    returned = scraper.scrape('https://www.everydayhealth.com/drugs/')
    assert returned == 0


def test_scrape_default_parameter():
    """
    Tests that, when calling the scrape function with a scraper with default parameters,
    the correct types of data are stored in the 'reviews' attribute and that the
    correct number of reviews are collected (more than 20, this proves that it's
    scraping multiple pages)
    """
    scraper = EverydayHealthScraper()
    scraper.scrape('https://www.everydayhealth.com/drugs/gabapentin/reviews')
    assert len(scraper.reviews) > 20
    data_collected = list(scraper.reviews[0].keys())
    assert len(data_collected) == 4
    assert 'comment' in data_collected
    assert 'rating' in data_collected
    assert 'date' in data_collected
    assert 'drug' in data_collected


def test_scrape_with_parameters():
    """
    Tests that, when calling the scrape function with a scraper of non-default parameters, the
    correct types of data are stored in the 'reviews' attribute
    """
    scraper = EverydayHealthScraper(collect_urls=True)
    scraper.scrape('https://www.everydayhealth.com/drugs/gabapentin/reviews')
    assert len(scraper.reviews) > 20
    data_collected = list(scraper.reviews[0].keys())
    assert len(data_collected) == 5
    assert 'url' in data_collected


def test_scrape_assert_title_error():
    """
    Tests that when the scrape function is called with an invalid url that does have a
    title, but the title is wrong (doesn't have the phrase 'Drug Reviews') that an AssertionError
    is raised and the function returns 0
    """
    scraper = EverydayHealthScraper()
    returned = scraper.scrape('https://www.everydayhealth.com/drugs/')
    assert returned == 0


def test_scrape_no_reviews():
    """
    Tests that the scrape function works for page with no reviews
    """
    scraper = EverydayHealthScraper()
    scraper.scrape('https://www.everydayhealth.com/drugs/triprolidine/reviews')
    assert len(scraper.reviews) == 0


def test_get_url_real_drug_name():
    """
    Tests that the get_url function returns the correct url for a standard drug name ('dupixent')
    """
    scraper = EverydayHealthScraper()
    url = scraper.get_url('dupixent')
    assert url == 'https://www.everydayhealth.com/drugs/dupixent/reviews'


def test_url_fake_drug_name():
    """
    Tests tha the get_url function returns 'None' for a drug name that does not have a review
    page
    """
    scraper = EverydayHealthScraper()
    url = scraper.get_url('garbage')
    assert not url


def test_drug_name_with_space():
    """
    Tests that the get_url function returns the correct url for a drug name
    that includes a space
    """
    scraper = EverydayHealthScraper()
    url = scraper.get_url('Mucinex Allergy')
    assert url == 'https://www.everydayhealth.com/drugs/mucinex-allergy/reviews'


def test_short_drug_name():
    """
    Tests that the get_url function does not search for drug names shorter than 4 characters
    """
    scraper = EverydayHealthScraper()
    url = scraper.get_url('ACE')
    assert not url





