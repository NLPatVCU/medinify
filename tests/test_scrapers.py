from medinify.scrapers import WebMDScraper
from medinify.scrapers import DrugsScraper

def test_webmd_max_pages():
    input_url = 'https://www.webmd.com/drugs/drugreview-151652-banzel.aspx?drugid=151652&drugname=banzel'
    webmd_scraper = WebMDScraper()
    assert webmd_scraper.max_pages(input_url) == 2

def test_webmd_scrape_page():
    input_url = 'https://www.webmd.com/drugs/drugreview-1701-citalopram-oral.aspx?drugid=1701&drugname=citalopram-oral'
    webmd_scraper = WebMDScraper()
    webmd_scraper.scrape_page(input_url)
    assert webmd_scraper.review_list
    assert 'comment' and 'effectiveness' and 'satisfaction' in webmd_scraper.review_list[-1]

def test_webmd_scrape():
    input_url = 'https://www.webmd.com/drugs/drugreview-151652-banzel.aspx?drugid=151652&drugname=banzel'
    webmd_scraper = WebMDScraper()
    webmd_scraper.scrape(input_url)
    assert len(webmd_scraper.review_list) > 5
    assert 'comment' and 'effectiveness' and 'satisfaction' in webmd_scraper.review_list[-1]

def test_drugs_max_pages():
    input_url = 'https://www.drugs.com/comments/dabigatran/'
    drugs_scraper = DrugsScraper()
    assert drugs_scraper.max_pages(input_url) == 2

def test_drugs_scrape():
    input_url = 'https://www.drugs.com/comments/dabigatran/'
    drugs_scraper = DrugsScraper()
    drugs_scraper.scrape(input_url, 'dabigatran.csv', 4)
    assert len(drugs_scraper.review_list) > 5
    assert 'comment' and 'for' and 'rating' in  drugs_scraper.review_list[-1]