"""
Examples for how to use the Medinify package
"""

from medinify.scrapers import WebMDScraper
from medinify.scrapers import DrugsScraper

def main():
    """ Main function.
    """
    input_url = 'https://www.webmd.com/drugs/drugreview-1701-citalopram-oral.aspx?drugid=1701&drugname=citalopram-oral'
    webmd_scraper = WebMDScraper()
    webmd_scraper.scrape(input_url, 'test.csv')

    # drugs_url = 'https://www.drugs.com/comments/dabigatran/'
    # drugs_scraper = DrugsScraper()
    # drugs_scraper.scrape(drugs_url, 'dabigatran.csv', 4)

if __name__ == "__main__":
    main()
