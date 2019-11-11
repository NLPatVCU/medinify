

import pytest
from medinify.scrapers import DrugRatingzScraper


def test_default_initialization():
    """
    Tests that when the new scraper object is initialized, the correct
    attributes are set (empty list[dict] 'reviews') and the attribute
    collect_urls (boolean) is false by default
    """
    scraper = DrugRatingzScraper()
    assert len(scraper.reviews) == 0
    assert not scraper.collect_urls


def test_initialization_parameters():
    """
    Tests that when a new DrugRatingz scraper object is initialized with
    non default arguments the correct attributes are set (empty list[dict] 'reviews')
    and that the collect_urls attribute (boolean) is set to true
    """
    scraper = DrugRatingzScraper(collect_urls=True)
    assert scraper.collect_urls


def test_incorrect_url():
    """
    Tests that the scrape_page function raises and assertion error when a url is
    entered with an incorrect domain name (not 'https://drugratingz.com...')
    """
    scraper = DrugRatingzScraper()
    with pytest.raises(AssertionError):
        scraper.scrape_page('https://www.drugs.com/comments/aripiprazole/abilify.html?page=1')


def test_no_reviews():
    """
    Tests that the scrape page function returns 'None' when no reviews are
    found on the page
    """
    scraper = DrugRatingzScraper()
    returned = scraper.scrape_page('https://www.drugratingz.com/reviews/18088/Drug-Affinitor.html')
    assert not returned


def test_scrape_correct_review_data():
    """
    Tests to make sure that the last review in the scraped reviews list has
    the correct data when the scrape function is called
    (this data is from the oldest review of the drug)
    """
    scraper = DrugRatingzScraper(collect_urls=True)
    scraper.scrape('https://www.drugratingz.com/reviews/141/Drug-Prednisone.html')
    assert scraper.reviews[-1]['comment'][:10] == 'This is a '
    assert scraper.reviews[-1]['comment'][-10:] == 'ing on it.'
    assert scraper.reviews[-1]['rating']['effectiveness'] == 5
    assert scraper.reviews[-1]['rating']['no side effects'] == 1
    assert scraper.reviews[-1]['rating']['convenience'] == 1
    assert scraper.reviews[-1]['rating']['value'] == 3
    assert scraper.reviews[-1]['date'] == '8/18/05'


def test_scrape_page_default_parameters():
    """
    Tests to make sure that calling the scrape_page function on a scraper object
    with a default parameter collects the correct types of data ('comment', 'rating',
    'date', and 'drug') and that the correct number of reviews were collected (13)
    """
    scraper = DrugRatingzScraper()
    scraper.scrape_page('https://www.drugratingz.com/reviews/141/Drug-Prednisone.html')
    data_collected = list(scraper.reviews[0].keys())
    assert len(data_collected) == 4
    assert 'comment' in data_collected
    assert 'rating' in data_collected
    assert 'date' in data_collected
    assert 'drug' in data_collected
    assert len(scraper.reviews) == 13


def test_scrape_page_with_parameters():
    """
    Tests to make sure that calling the scrape_page function on a scraper object with a
    non-default parameter (collect_url true) collects the correct types of data
    ('comment', 'rating, 'date', 'drug', and 'url')
    """
    scraper = DrugRatingzScraper(collect_urls=True)
    scraper.scrape_page('https://www.drugratingz.com/reviews/141/Drug-Prednisone.html')
    data_collected = list(scraper.reviews[0].keys())
    assert len(data_collected) == 5
    assert 'url' in data_collected


def test_scrape_empty_reviews():
    """
    Tests to make sure that the scrape function would discard the reviews
    of a scraper object that already has data collected in 'reviews'
    """
    scraper = DrugRatingzScraper()
    scraper.scrape('https://www.drugratingz.com/reviews/258/Drug-Glucophage.html')
    num_reviews = len(scraper.reviews)
    scraper.scrape('https://www.drugratingz.com/reviews/258/Drug-Glucophage.html')
    assert num_reviews == len(scraper.reviews)


def test_scrape_invalid_url_no_title():
    """
    Tests that when the scrape function is called on a url that lacks a title
    (invalid url), it raises an AttributeError and returns 'None'
    """
    scraper = DrugRatingzScraper()
    returned = scraper.scrape('https://www.drugratingz.com/reviews/258/Drug.html')
    assert not returned


def test_scrape_with_parameters():
    """
    Tests that, when calling the scrape function with a scraper of non-default parameters, the
    correct types of data are stored in the 'reviews' attribute
    """
    scraper = DrugRatingzScraper(collect_urls=True)
    scraper.scrape('https://www.drugratingz.com/reviews/472/Drug-Lutera.html')
    assert len(scraper.reviews) > 10
    data_collected = list(scraper.reviews[0].keys())
    assert len(data_collected) == 5
    assert 'url' in data_collected


def test_scrape_assert_title_error():
    """
    Tests that when the scrape function is called with an invalid url that does have a
    title, but the title is wrong (doesn't have the phrase 'drug reviews') that an AssertionError
    is raised and the function returns 0
    """
    scraper = DrugRatingzScraper()
    returned = scraper.scrape('https://www.drugratingz.com/ShowThingCats.jsp')
    assert returned == 0


def test_scrape_no_reviews():
    """
    Tests that the scrape function works for page with no reviews
    """
    scraper = DrugRatingzScraper()
    scraper.scrape('https://www.drugratingz.com/reviews/18589/Drug-Adderall.html')
    assert len(scraper.reviews) == 0


def test_get_url_real_drug_name():
    """
    Tests that the get_url function returns the correct url for a standard drug name ('actos')
    """
    scraper = DrugRatingzScraper()
    url = scraper.get_url('actos')
    assert url == 'https://www.drugratingz.com/reviews/340/Drug-Actos.html'


def test_url_fake_drug_name():
    """
    Tests tha the get_url function returns 'None' for a drug name that does not have a review
    page
    """
    scraper = DrugRatingzScraper()
    url = scraper.get_url('garbage')
    assert not url


def test_drug_name_with_space():
    """
    Tests that the get_url function returns the correct url for a drug name
    that includes a space
    """
    scraper = DrugRatingzScraper()
    url = scraper.get_url('Ortho Tri-Cyclen Lo ')
    assert url == 'https://www.drugratingz.com/reviews/163/Drug-Ortho-Tri-Cyclen-Lo.html'


def test_short_drug_name():
    """
    Tests that the get_url function does not search for drug names shorter than 4 characters
    """
    scraper = DrugRatingzScraper()
    url = scraper.get_url('ACE')
    assert not url



