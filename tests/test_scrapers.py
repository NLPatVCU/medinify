from medinify.scrapers import WebMDScraper


def test_webmd_max_pages():
    input_url = 'https://www.webmd.com/drugs/drugreview-151652-banzel.aspx?drugid=151652&drugname=banzel'
    webmd_scraper = WebMDScraper()
    assert webmd_scraper.max_pages(input_url) == 2

def test_webmd_scrape_page():
    input_url = 'https://www.webmd.com/drugs/drugreview-1701-citalopram-oral.aspx?drugid=1701&drugname=citalopram-oral'
    webmd_scraper = WebMDScraper()
    webmd_scraper.scrape_page(input_url)
    assert len(webmd_scraper.review_list) > 0
    assert 'comment' and 'effectiveness' and 'satisfaction' in webmd_scraper.review_list[-1]

def test_webmd_scrape():
    input_url = 'https://www.webmd.com/drugs/drugreview-151652-banzel.aspx?drugid=151652&drugname=banzel'
    webmd_scraper = WebMDScraper()
    webmd_scraper.scrape(input_url)
    assert len(webmd_scraper.review_list) > 5
    assert 'comment' and 'effectiveness' and 'satisfaction' in webmd_scraper.review_list[-1]
