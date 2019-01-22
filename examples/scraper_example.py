"""
Examples for how to use the Medinify package
"""

from medinify.scrapers import WebMDScraper
from medinify.scrapers import DrugsScraper

def main():
    """ Main function.
    """
    # input_url = 'https://www.webmd.com/drugs/drugreview-12120-Doxil+intravenous.aspx?drugid=12120&drugname=Doxil+intravenous&pageIndex=1&sortby=3&conditionFilter=-1'
    # webmd_scraper = WebMDScraper("citalopram_train.csv")
    # webmd_scraper.scrape(input_url, 10)

    drugs_url = 'https://www.drugs.com/comments/dabigatran/'
    drugs_scraper = DrugsScraper()
    drugs_scraper.scrape(drugs_url, 'dabigatran.csv', 4)

if __name__ == "__main__":
    main()
