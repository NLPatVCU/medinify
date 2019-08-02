"""
Examples for how to use the Medinify package
"""

from medinify.scrapers import WebMDScraper, DrugRatingzScraper, DrugsScraper, EverydayHealthScraper


def main():
    """ Main function.
    """
    input_url = 'https://www.webmd.com/drugs/drugreview-1701-citalopram-oral.aspx?drugid=1701&drugname=citalopram-oral'
    webmd_scraper = WebMDScraper()
    webmd_scraper.scrape(input_url)


if __name__ == "__main__":
    main()
