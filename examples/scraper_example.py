"""
Examples for how to use the Medinify package
"""

from medinify.scrapers import WebMDScraper
#from medinify.scrapers import DrugsScraper
#from medinify.scrapers import IodineScraper 

def main():
    """ Main function.
    """
    input_url = 'https://www.webmd.com/drugs/drugreview-1701-citalopram-oral.aspx?drugid=1701&drugname=citalopram-oral'
    webmd_scraper = WebMDScraper()
    webmd_scraper.scrape(input_url)

    # drugs_url = 'https://www.drugs.com/comments/dabigatran/'
    # drugs_scraper = DrugsScraper()
    # drugs_scraper.scrape(drugs_url, 'dabigatran.csv', 4)

    #iodine_url = "https://www.iodine.com/drug/adderall/reviews"
    #iodine_scraper = IodineScraper()
    #iodine_scraper.scraper(iodine_url, 'test.csv')

if __name__ == "__main__":
    main()
