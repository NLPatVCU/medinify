"""
Tests for all drug review scrapers
"""

import os
from medinify.scrapers import WebMDScraper
from medinify.scrapers import IodineScraper
from medinify.scrapers import DrugRatingzScraper
from medinify.scrapers import DrugsScraper
from medinify.scrapers import EverydayHealthScraper


def test_webmd_max_pages():
    """Test webmd max pages"""
    input_url = 'https://www.webmd.com/drugs/drugreview-151652-banzel.aspx?drugid=151652&drugname=banzel'
    webmd_scraper = WebMDScraper()
    assert webmd_scraper.max_pages(input_url) == 3


def test_webmd_scrape_page():
    """Test webmd scrape page"""
    input_url = 'https://www.webmd.com/drugs/drugreview-1701-citalopram-oral.aspx?drugid=1701&drugname=citalopram-oral'
    webmd_scraper = WebMDScraper()
    webmd_scraper.scrape_page(input_url)
    assert webmd_scraper.review_list

    keys = list(webmd_scraper.review_list[-1].keys())
    assert 'comment' in keys
    assert 'effectiveness' in keys
    assert 'ease of use' in keys
    assert 'satisfaction' in keys


def test_webmd_scrape():
    """Test webmd scrape"""
    input_url = 'https://www.webmd.com/drugs/drugreview-151652-banzel.aspx?drugid=151652&drugname=banzel'
    webmd_scraper = WebMDScraper()
    webmd_scraper.scrape(input_url)
    assert len(webmd_scraper.review_list) > 5

    keys = list(webmd_scraper.review_list[-1].keys())
    assert 'comment' in keys
    assert 'effectiveness' in keys
    assert 'ease of use' in keys
    assert 'satisfaction' in keys


def test_iodine_scrape():
    """Test iodine scrape"""
    input_url = 'https://www.iodine.com/drug/adderall/reviews'
    iodine_scraper = IodineScraper()
    iodine_scraper.scraper(input_url, 'test.csv')
    assert os.path.exists('test.csv')
    os.remove('test.csv')


def test_drugratingz_scrape():
    """Test drug ratingz scrape"""
    url = 'https://www.drugratingz.com/reviews/75/Drug-Adderall-XR.html'
    drug_scraper = DrugRatingzScraper()
    drug_scraper.scrape(url, 'test.csv')
    assert os.path.exists('test.csv')
    os.remove('test.csv')


def test_drugs_scrape():
    """Test drugs scrape"""
    url = 'https://www.drugs.com/comments/dabigatran/'
    drugs_scraper = DrugsScraper()
    drugs_scraper.scrape(url, 'test.csv', pages=1)
    assert os.path.exists('test.csv')
    os.remove('test.csv')

# def test_get_drug_urls():
#    scraper = WebMDScraper()
#    scraper.get_drug_urls('test-drug-names.csv', 'test-drug-urls.csv')
#    assert os.path.exists('test-drug-urls.csv')
#    os.remove('test-drug-urls.csv')

def test_everydayhealth_max_pages():
    """Test everydayhealth max pages"""
    url = 'https://www.everydayhealth.com/drugs/citalopram/reviews'
    everydayhealth_scraper = EverydayHealthScraper()
    assert everydayhealth_scraper.max_pages(url) == 15

def test_everydayhealth_scrape():
    url = 'https://www.everydayhealth.com/drugs/citalopram/reviews'
    everydayhealth_scraper = EverydayHealthScraper()
    everydayhealth_scraper.scrape(url, 'test.csv', 4)
    assert os.path.exists('test.csv')
    os.remove('test.csv')
