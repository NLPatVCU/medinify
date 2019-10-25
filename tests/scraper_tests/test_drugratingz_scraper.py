
# TODO (Dunya) - Add tests for DrugRatingz Scraper

import pytest
from medinify.scrapers import DrugRatingzScraper


def test_default_initialization():
    scraper = DrugRatingzScraper()
    assert len(scraper.reviews) == 0
    assert not scraper.collect_urls

def test_initialization_parameters():
    scraper = DrugRatingzScraper(collect_urls=True)
    assert scraper.collect_urls

def test_incorrect_url():
    scraper = DrugRatingzScraper()
    with pytest.raises(AssertionError):
        scraper.scrape_page('https://www.drugs.com/comments/aripiprazole/abilify.html?page=1')

def test_no_reviews():
    scraper = DrugRatingzScraper()
    returned = scraper.scrape_page(
        'https://www.webmd.com/drugs/drugreview-155251-T-Plus-topical.aspx?drugid=155251&drugname=T-Plus-topical')
    assert returned == 0

def test_scrape_correct_review_data():
    scraper = DrugRatingzScraper(collect_user_ids=True)

