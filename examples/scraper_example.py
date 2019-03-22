"""
Examples for how to use the Medinify package
"""

#from medinify.scrapers import WebMDScraper
#from medinify.scrapers import DrugsScraper
# from medinify.scrapers import IodineScraper 
from medinify.scrapers import EverydayHealthScraper
#from medinify.scrapers import DrugRatingzScraper

def main():
    """ Main function.
    """
    # input_url = 'https://www.webmd.com/drugs/drugreview-1701-citalopram-oral.aspx?drugid=1701&drugname=citalopram-oral'
    # webmd_scraper = WebMDScraper()
    # webmd_scraper.scrape(input_url)

    # input_url = 'https://www.drugs.com/comments/dabigatran/'
    # drugs_scraper = DrugsScraper()
    # drugs_scraper.scrape(drugs_url, 'dabigatran.csv', 4)

    # iodine_url = "https://www.iodine.com/drug/adderall/reviews"
    # iodine_scraper = IodineScraper()
    # iodine_scraper.scraper(iodine_url, 'adderall.csv')

    input_url = 'https://www.everydayhealth.com/drugs/citalopram/reviews'
    everydayhealth_scraper = EverydayHealthScraper()
    everydayhealth_scraper.max_pages(input_url)
    everydayhealth_scraper.scrape(input_url, 'citalopram.csv', 2)
    

    #input_url = 'https://www.drugratingz.com/reviews/75/Drug-Adderall-XR.html'
    #drugratingz_scraper = DrugRatingzScraper()
    #drugratingz_scraper.scrape(input_url, 'adderallxr.csv')



if __name__ == "__main__":
    main()
