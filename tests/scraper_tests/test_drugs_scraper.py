
import pytest
from medinify.scrapers import DrugsScraper


def test_default_initialization():
    """
    Test that when a new scraper object is initialized, the correct
    attributes are set (empty list[dict] 'reviews') and that collect_urls
    and collect_user_ids attributes (Booleans) are false by default
    """
    scraper = DrugsScraper()
    assert len(scraper.reviews) == 0
    assert not scraper.collect_urls
    assert not scraper.collect_user_ids


def test_initialization_parameters():
    """
    Test that when a new scraper object is initialized with non-default arguments
    the correct attributes are set (empty list[dict] 'reviews') and that collect_urls
    and collect_user_ids attributes (Booleans) are set true
    """
    scraper = DrugsScraper(collect_urls=True, collect_user_ids=True)
    assert len(scraper.reviews) == 0
    assert scraper.collect_urls
    assert scraper.collect_user_ids


def test_scrape_page_incorrect_website():
    """
    Tests that the scrape_page function raises an assertion error
    when a url is entered with an incorrect domain name (not 'https://drugs.com...')
    """
    scraper = DrugsScraper()
    with pytest.raises(AssertionError):
        scraper.scrape_page('https://www.webmd.com/drugs/drugreview-64439-abilify.aspx?drugid=64439&drugname=abilify')


def test_scrape_page_no_reviews():
    """
    Tests that the scrape page function returns 0 when
    no reviews are found on the page
    """
    scraper = DrugsScraper()
    returned = scraper.scrape_page('https://www.drugs.com/comments/xylitol/')
    assert returned == 0


def test_scrape_page_default_parameters():
    """
    Tests to make sure that calling the scrape_page function
    on a scraper object with default parameters collects the
    correct types of data ('comment', 'rating', 'date', and 'drug')
    and that the correct number of reviews (25) were collected
    """
    scraper = DrugsScraper()
    scraper.scrape_page('https://www.drugs.com/comments/acetaminophen/?page=2')
    data_collected = list(scraper.reviews[0].keys())
    assert len(data_collected) == 4
    assert 'comment' in data_collected
    assert 'rating' in data_collected
    assert 'date' in data_collected
    assert 'drug' in data_collected
    assert len(scraper.reviews) == 25


def test_scrape_page_with_parameters():
    """
    Tests to make sure that calling the scrape_page function
    on a scraper object with non-default parameters (collect_url
    and collect_user_id true) collects the correct types of
    data ('comment', 'rating', 'date', 'drug', 'url', and 'user id')
    """
    scraper = DrugsScraper(collect_user_ids=True, collect_urls=True)
    scraper.scrape_page('https://www.drugs.com/comments/acetaminophen/?page=2')
    data_collected = list(scraper.reviews[0].keys())
    assert len(data_collected) == 6
    assert 'user id' in data_collected
    assert 'url' in data_collected


def test_scrape_empty_reviews():
    """
    Tests to make sure that if the scrape function is called on a scraper
    that already has collected data in 'reviews', that those reviews are discarded
    """
    scraper = DrugsScraper()
    scraper.scrape('https://www.drugs.com/comments/acetaminophen/')
    num_reviews = len(scraper.reviews)
    scraper.scrape('https://www.drugs.com/comments/acetaminophen/')
    assert num_reviews == len(scraper.reviews)


def test_scrape_correct_review_data():
    """
    Tests to make sure that when the scrape function is called,
    that the last review in the scraped reviews list as the correct data
    (this data is the data from the oldest review for this drug)
    """
    scraper = DrugsScraper(collect_user_ids=True)
    scraper.scrape('https://www.drugs.com/comments/acetaminophen/')
    assert scraper.reviews[-1]['comment'][:11] == 'I only have'
    assert scraper.reviews[-1]['comment'][-10:] == ' in check.'
    assert scraper.reviews[-1]['rating'] == 10.0
    assert scraper.reviews[-1]['drug'] == 'Acetaminophen'
    assert scraper.reviews[-1]['date'] == 'March 4, 2008'
    assert scraper.reviews[-1]['user id'] == 'Anonymous'


def test_scrape_invalid_url_no_title():
    """
    Tests that when the scrape function is called on a url that
    lacks a title (invalid url), it raises an AttributeError and returns 0
    """
    scraper = DrugsScraper()
    returned = scraper.scrape('https://www.webmd.com/drugs/drugreview-bhilknhj')
    assert returned == 0


def test_scrape_wrong_title_url():
    """
    Tests that when the scrape function is called with an invalid url
    that does have a title but the title is wrong (doesn't have the phrase 'User Reviews for')
    that an AssertionError is raised and the function returns 0
    """
    scraper = DrugsScraper()
    returned = scraper.scrape('https://www.drugs.com/comments/fghjkjhgdfgh/')
    assert returned == 0


def test_scrape_default_parameter():
    """
    Tests that, when calling scrape function with a scraper with default parameters
    the correct types of data are stored in the 'reviews' attribute
    and that the correct number of reviews are collected (more
    than 25, this proves that it's scraping multiple pages)
    """
    scraper = DrugsScraper()
    scraper.scrape('https://www.drugs.com/comments/acetaminophen/')
    assert len(scraper.reviews) > 25
    data_collected = list(scraper.reviews[0].keys())
    assert len(data_collected) == 4
    assert 'comment' in data_collected
    assert 'rating' in data_collected
    assert 'date' in data_collected
    assert 'drug' in data_collected


def test_scrape_with_parameters():
    """
    Tests that, when calling scrape function with a scraper with non-default parameters
    the correct types of data are stored in the 'reviews' attribute
    """
    scraper = DrugsScraper(collect_urls=True, collect_user_ids=True)
    scraper.scrape('https://www.drugs.com/comments/acetaminophen/')
    data_collected = list(scraper.reviews[0].keys())
    assert len(data_collected) == 6
    assert 'user id' in data_collected
    assert 'url' in data_collected


def test_get_url_real_drug_name():
    """
    Tests that the get_url function returns the correct url for a standard drug name ('actos')
    """
    scraper = DrugsScraper()
    url = scraper.get_url('actos')
    assert url == 'https://www.drugs.com/comments/pioglitazone/actos.html'


def test_url_fake_drug_name():
    """
    Tests that the get_url function returns 'None' for a drug name that does not have a review page
    """
    scraper = DrugsScraper()
    url = scraper.get_url('garbage')
    assert not url


def test_drug_name_with_space():
    """
    Tests that the get_url function returns the correct url for a drug name with a space in it
    """
    scraper = DrugsScraper()
    url = scraper.get_url('Advair Diskus')
    assert url == 'https://www.drugs.com/comments/fluticasone-salmeterol/advair-diskus.html'


def test_short_drug_name():
    """
    Tests that the get_url function does not search for drug names shorter than 4 characters
    """
    scraper = DrugsScraper()
    url = scraper.get_url('ACE')
    assert not url


def test_name_with_numbers():
    """
    Tests get_url function on drug name with numbers
    """
    scraper = DrugsScraper()
    url = scraper.get_url('B3-500-Gr')
    assert url == 'https://www.drugs.com/comments/niacin/b3-500-gr.html'


def test_scrape_no_reviews():
    """
    Tests that scrape function works for page with no reviews
    """
    scraper = DrugsScraper()
    scraper.scrape('https://www.drugs.com/comments/xylitol/')
    assert len(scraper.reviews) == 0

