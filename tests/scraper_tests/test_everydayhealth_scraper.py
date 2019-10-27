
# TODO (Dunya) - Add tests for EverydayHealth Scraper

import pytest
from medinify.scrapers import EverydayHealthScraper


def test_default_initialization():
    scraper = EverydayHealthScraper()
    assert len(scraper.reviews) == 0
    assert not scraper.collect_urls


def test_initialization_parameters():
    scraper = EverydayHealthScraper(collect_urls=True)
    assert scraper.collect_urls


def test_incorrect_url():
    scraper = EverydayHealthScraper()
    with pytest.raises(AssertionError):
        scraper.scrape_page('https://www.drugs.com/comments/aripiprazole/abilify.html?page=1')


def test_no_reviews():
    scraper = EverydayHealthScraper()
    returned = scraper.scrape_page('https://www.everydayhealth.com/drugs/triprolidine/reviews')
    assert returned is None


def test_scrape_correct_review_data():
    scraper = EverydayHealthScraper(collect_urls=True)
    scraper.scrape('https://www.everydayhealth.com/drugs/ciclopirox-topical/reviews')
    assert scraper.reviews[-1]['comment'][:10] == 'After OVER'
    assert scraper.reviews[-1]['comment'][-10:] == 'inally hav'
    assert scraper.reviews[-1]['rating'] == 5
    assert scraper.reviews[-1]['date'] == '5/22/2015 4:18:19 AM'


def test_scrape_page_default_parameters():
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
    scraper = EverydayHealthScraper(collect_urls=True)
    scraper.scrape_page('https://www.everydayhealth.com/drugs/hydroxyzine/reviews')
    data_collected = list(scraper.reviews[0].keys())
    assert len(data_collected) == 5
    assert 'url' in data_collected


def test_scrape_empty_reviews():
    scraper = EverydayHealthScraper()
    scraper.scrape('https://www.everydayhealth.com/drugs/phenadoz/reviews')
    num_reviews = len(scraper.reviews)
    scraper.scrape('https://www.everydayhealth.com/drugs/phenadoz/reviews')
    assert num_reviews == len(scraper.reviews)


def test_scrape_invalid_url_no_title():
    scraper = EverydayHealthScraper()
    returned = scraper.scrape('https://www.everydayhealth.com/drugs/')
    assert returned == 0


def test_scrape_default_parameter():
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
    scraper = EverydayHealthScraper(collect_urls=True)
    scraper.scrape('https://www.everydayhealth.com/drugs/gabapentin/reviews')
    assert len(scraper.reviews) > 20
    data_collected = list(scraper.reviews[0].keys())
    assert len(data_collected) == 5
    assert 'url' in data_collected


def test_scrape_assert_title_error():
    scraper = EverydayHealthScraper()
    returned = scraper.scrape('https://www.everydayhealth.com/drugs/')
    assert returned == 0


def test_scrape_no_reviews():
    scraper = EverydayHealthScraper()
    scraper.scrape('https://www.everydayhealth.com/drugs/triprolidine/reviews')
    assert len(scraper.reviews) == 0


def test_get_url_real_drug_name():
    scraper = EverydayHealthScraper()
    url = scraper.get_url('dupixent')
    assert url == 'https://www.everydayhealth.com/drugs/dupixent/reviews'


def test_url_fake_drug_name():
    scraper = EverydayHealthScraper()
    url = scraper.get_url('garbage')
    assert not url


def test_drug_name_with_space():
    scraper = EverydayHealthScraper()
    url = scraper.get_url('Mucinex Allergy')
    assert url == 'https://www.everydayhealth.com/drugs/mucinex-allergy/reviews'


def test_short_drug_name():
    scraper = EverydayHealthScraper()
    url = scraper.get_url('ACE')
    assert not url


def test_name_with_numbers():
    scraper = EverydayHealthScraper()
    url = scraper.get_url('5-htp')
    assert url == 'https://www.everydayhealth.com/drugs/5-htp-5-hydroxytryptophan/reviews'



