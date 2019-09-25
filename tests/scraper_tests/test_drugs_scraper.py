
import pytest
from medinify.scrapers import DrugsScraper


def test_default_initialization():
    scraper = DrugsScraper()
    assert len(scraper.reviews) == 0
    assert not scraper.collect_urls
    assert not scraper.collect_user_ids


def test_initialization_parameters():
    scraper = DrugsScraper(collect_urls=True, collect_user_ids=True)
    assert len(scraper.reviews) == 0
    assert scraper.collect_urls
    assert scraper.collect_user_ids


def test_scrape_page_incorrect_website():
    scraper = DrugsScraper()
    with pytest.raises(AssertionError):
        scraper.scrape_page('https://www.webmd.com/drugs/drugreview-64439-abilify.aspx?drugid=64439&drugname=abilify')


def test_scrape_page_no_reviews():
    scraper = DrugsScraper()
    returned = scraper.scrape_page('https://www.drugs.com/comments/xylitol/')
    assert returned == 0


def test_scrape_correct_review_data():
    scraper = DrugsScraper(collect_user_ids=True)
    scraper.scrape('https://www.drugs.com/comments/acetaminophen/')
    assert scraper.reviews[-1]['comment'][:11] == 'I only have'
    assert scraper.reviews[-1]['comment'][-10:] == ' in check.'
    assert scraper.reviews[-1]['rating'] == 10.0
    assert scraper.reviews[-1]['date'] == 'March 4, 2008'
    assert scraper.reviews[-1]['user id'] == 'Anonymous'


def test_scrape_page_default_parameters():
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
    scraper = DrugsScraper(collect_user_ids=True, collect_urls=True)
    scraper.scrape_page('https://www.drugs.com/comments/acetaminophen/?page=2')
    data_collected = list(scraper.reviews[0].keys())
    assert len(data_collected) == 6
    assert 'user id' in data_collected
    assert 'url' in data_collected
    assert len(scraper.reviews) == 25


def test_scrape_empty_reviews():
    scraper = DrugsScraper()
    scraper.scrape('https://www.drugs.com/comments/acetaminophen/')
    num_reviews = len(scraper.reviews)
    scraper.scrape('https://www.drugs.com/comments/acetaminophen/')
    assert num_reviews == len(scraper.reviews)


def test_scrape_invalid_url_no_title():
    scraper = DrugsScraper()
    returned = scraper.scrape('https://www.webmd.com/drugs/drugreview-bhilknhj')
    assert returned == 0


def test_scrape_wrong_title_url():
    scraper = DrugsScraper()
    returned = scraper.scrape('https://www.drugs.com/comments/fghjkjhgdfgh/')
    assert returned == 0


def test_scrape_default_parameter():
    scraper = DrugsScraper()
    scraper.scrape('https://www.drugs.com/comments/acetaminophen/')
    assert len(scraper.reviews) > 5
    data_collected = list(scraper.reviews[0].keys())
    assert len(data_collected) == 4
    assert 'comment' in data_collected
    assert 'rating' in data_collected
    assert 'date' in data_collected
    assert 'drug' in data_collected


def test_scrape_with_parameters():
    scraper = DrugsScraper(collect_urls=True, collect_user_ids=True)
    scraper.scrape('https://www.drugs.com/comments/acetaminophen/')
    assert len(scraper.reviews) > 5
    data_collected = list(scraper.reviews[0].keys())
    assert len(data_collected) == 6
    assert 'user id' in data_collected
    assert 'url' in data_collected


def test_get_url_real_drug_name():
    scraper = DrugsScraper()
    url = scraper.get_url('actos')
    assert url == 'https://www.drugs.com/comments/pioglitazone/actos.html'


def test_url_fake_drug_name():
    scraper = DrugsScraper()
    url = scraper.get_url('garbage')
    assert not url


def test_drug_name_with_space():
    scraper = DrugsScraper()
    url = scraper.get_url('Advair Diskus')
    assert url == 'https://www.drugs.com/comments/fluticasone-salmeterol/advair-diskus.html'


def test_short_drug_name():
    scraper = DrugsScraper()
    url = scraper.get_url('ACE')
    assert not url


def test_name_with_numbers():
    scraper = DrugsScraper()
    url = scraper.get_url('B3-500-Gr')
    assert url == 'https://www.drugs.com/comments/niacin/b3-500-gr.html'


def test_get_url_inexact_name():
    scraper = DrugsScraper()
    url = scraper.get_url('tylenol')
    assert url == 'https://www.drugs.com/comments/acetaminophen/tylenol.html'


def test_scrape_no_reviews():
    scraper = DrugsScraper()
    scraper.scrape('https://www.drugs.com/comments/xylitol/')
    assert len(scraper.reviews) == 0

