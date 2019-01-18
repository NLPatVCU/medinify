"""
Examples for how to use the Medinify package
"""

from medinify.scrapers import WebMDScraper
from medinify.scrapers import RXListScraper

def main():
    """ Main function.
    """

    input_url = "https://www.webmd.com/drugs/drugreview-12120-Doxil+intravenous.aspx?drugid=12120&drugname=Doxil+intravenous"
    scraper = WebMDScraper("webmd_doxil.csv")
    scraper.scrape(input_url, 10)

    # rxlist_scraper = RXListScraper()
    # rxlist_scraper.scrape('https://www.rxlist.com/script/main/rxlist_view_comments.asp?drug=visudyne&questionid=fdb18177_pem', 'rxlist_visudyne.csv')


if __name__ == "__main__":
    main()
