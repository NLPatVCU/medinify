"""
Tests for all drug review scrapers
"""

import os
from medinify.scrapers import WebMDScraper
from medinify.scrapers import IodineScraper
from medinify.scrapers import DrugRatingzScraper
from medinify.scrapers import DrugsScraper

def test_webmd_max_pages():
    """Test webmd max pages"""
    input_url = 'https://www.webmd.com/drugs/drugreview-151652-banzel.aspx?drugid=151652&drugname=banzel'
    webmd_scraper = WebMDScraper()
    assert webmd_scraper.max_pages(input_url) == 2


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

# TODO (Jorge): Fix drugratingz scraper. This test is correctly failing.
# def test_drugratingz_scrape():
#     """Test drug ratingz scrape"""
#     url = 'https://www.drugratingz.com/reviews/75/Drug-Adderall-XR.html'
#     drug_scraper = DrugRatingzScraper()
#     drug_scraper.scrape(url, 'test.csv')
#     assert os.path.exists('test.csv')
#     os.remove('test.csv')

def test_drugs_max_pages():
    """Test drugs.com max pages"""
    input_url = 'https://www.drugs.com/comments/dabigatran/'
    drugs_scraper = DrugsScraper()
    assert drugs_scraper.max_pages(input_url) > 1
    
def test_drugs_scrape():
    """Test drugs.com scrape"""
    url = 'https://www.drugs.com/comments/dabigatran/'
    drugs_scraper = DrugsScraper()
    drugs_scraper.scrape(url, 1)
    assert len(drugs_scraper.review_list) > 5
