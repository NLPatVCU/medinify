"""
Examples for how to use the Medinify package
"""

from medinify.scrapers import WebMDScraper
# from medinify.scrapers import RXListScraper

def main():
    """ Main function.
    """

    input_url = "https://www.webmd.com/drugs/drugreview-1701-citalopram-oral.aspx?drugid=1701&drugname=citalopram-oral"
    scraper = WebMDScraper("citalopram_train.csv")
    scraper.scrape(input_url, 10)

    # rxlist_scraper = RXListScraper()
    # rxlist_scraper.scrape('https://www.rxlist.com/script/main/rxlist_view_comments.asp?drug=doxil&questionid=fdb12120_pem', 'doxil.csv')


if __name__ == "__main__":
    main()
