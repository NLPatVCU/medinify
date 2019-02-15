#from medinify.scrapers import WebMDScraper
from medinify.scrapers import IodineScraper
from medinify.scrapers import DrugRatingzScraper
from medinify.scrapers import DrugsScraper


'''
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
    '''

#Test for iodine.com 
def test_iodine_scrape(): 
    input_url = 'https://www.iodine.com/drug/adderall/reviews'
    iodine_scraper = IodineScraper()
    iodine_scraper.scraper(input_url)
    assert iodine_scraper.review_list
    assert 'comment' and 'worth it' and 'worked well' and 'big hassle' in iodine_scraper.review_list[-1]

#Test for drugratingz.com
#need to review 
def test_drugratingz_scrape():
    input = 'https://www.drugratingz.com/reviews/75/Drug-Adderall-XR.html'
    output_path = 'drugRatingz.csv' 
    drug_scraper = DrugRatingzScraper()
    drug_scraper.scrape(input, output_path)
    assert drug_scraper.review_list
    assert 'comment' and 'effectiveness' and 'no side effects' and 'convenience' and 'value' in drug_scraper.review_list[-1]

#test for drugs.com 
#need to review
def test_drugs_scrape():
    input = 'https://www.drugs.com/comments/dabigatran/'
    output_path = 'dabigatran.csv'
    drugs_scraper = DrugsScraper
    drugs_scraper.scrape(input, output_path, pages=1)
    assert drugs_scraper.review_list
    assert 'comment' and 'for' and 'rating' in drugs_scraper.review_list[-1]

#Test for everydayhealth.com 

