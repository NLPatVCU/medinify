"""
Example of how to use Medinify's scraping functionality
"""

from medinify.scrapers import WebMDScraper, DrugRatingzScraper, DrugsScraper, EverydayHealthScraper


def main():
    scraper = WebMDScraper()  # or DrugsScraper(), DrugRatingsScraper(), or EverydayHealthScraper()
    url = scraper.get_url('Galzin')  # or any other drug name
    scraper.scrape(url)
    print('Scraped %d reviews.' % len(scraper.reviews))


if __name__ == '__main__':
    main()

