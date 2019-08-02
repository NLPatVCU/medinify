"""
EverydayHealth.com drug review scraper
"""

import re
import requests
from bs4 import BeautifulSoup
from medinify.scrapers.scraper import Scraper
import pandas as pd


class EverydayHealthScraper(Scraper):
    """Scrapes EverydayHealth.com for drug reviews.
    """

    def __init__(self, collect_ratings=True, collect_dates=True, collect_drugs=True,
                 collect_user_ids=False, collect_urls=False):
        super(EverydayHealthScraper, self).__init__(collect_ratings, collect_dates,
                                                    collect_drugs, collect_user_ids,
                                                    collect_urls)
        if 'user id' in self.data_collected:
            raise AttributeError('EverydayHealth.com does not contain user id data')

    def scrape_page(self, url):
        """
        Scrapes a single page of drug reviews
        :param url: drug reviews page url
        """
        page = requests.get(url)
        soup = BeautifulSoup(page.text, 'html.parser')
        reviews = soup.find_all('div', {'itemprop': 'review'})
        drug_name = soup.find('title').text.split()[0]

        rows = {'comment': []}
        if 'rating' in self.data_collected:
            rows['rating'] = []
        if 'date' in self.data_collected:
            rows['date'] = []
        if 'drug' in self.data_collected:
            rows['drug'] = []
        if 'url' in self.data_collected:
            rows['url'] = []

        for review in reviews:
            comment = review.find('p', {'itemprop': 'reviewBody'}).text[:-7]
            if type(comment) == float:
                continue
            rows['comment'].append(comment)
            if 'rating' in self.data_collected:
                rating = None
                if review.find('span', {'itemprop': 'reviewRating'}):
                    rating = float(review.find('span', {'itemprop': 'reviewRating'}).text)
                rows['rating'].append(rating)
            if 'date' in self.data_collected:
                rows['date'].append(review.find('span', {'class': 'time'}).attrs['content'])
            if 'drug' in self.data_collected:
                rows['drug'].append(drug_name)
            if 'url' in self.data_collected:
                rows['url'].append(url)

        scraped_data = pd.DataFrame(rows, columns=self.data_collected)
        self.dataset = self.dataset.append(scraped_data, ignore_index=True)

    def scrape(self, url):
        """
        Scrapes all reviews of a given drug
        :param url: drug reviews url
        """
        print('Scraping WebMD...')
        pages = max_pages(url)

        for i in range(pages):
            page_url = url + '/' + str(i + 1)
            self.scrape_page(page_url)

            if (i + 1) % 10 == 0:
                print('Scraped {} of {} pages...'.format(i + 1, pages))

    def get_url(self, drug_name):
        """
        Given a drug name, finds the drug review page(s) on a given review forum
        :param drug_name: name of drug being searched for
        :return: drug url on given review forum
        """
        url = []
        drug = re.sub('\s+', '-', drug_name.lower())
        search_url = 'https://www.everydayhealth.com/drugs/' + drug + '/reviews'
        page = requests.get(search_url)
        search_soup = BeautifulSoup(page.text, 'html.parser')
        if 'Reviews' in search_soup.find('title').text.split():
            url.append(search_url)
        return url


def max_pages(input_url):
    """Finds number of review pages for this drug.
    Args:
        input_url: URL for the first page of reviews.
    Returns:
        (int) Highest page number
    """
    while True:
        page = requests.get(input_url)
        soup = BeautifulSoup(page.text, 'html.parser')

        # Case if no reviews available

        break
    if soup.find('div', {'class': 'review-details clearfix'}):
        total_reviews_head = soup.find('div', {'class': 'review-details clearfix'}).find('h5').find('span', {
            'itemprop': 'reviewCount'}).text
    else:
        return 0
    total_reviews = int(total_reviews_head)

    max_pages_foot = soup.find('div', {'class': 'review-pagination'}).find('section', {
        'class': 'review-pagination__section--info'}).text.split()
    max_pages = int(max_pages_foot[2])

    print('Found ' + str(total_reviews) + ' reviews.')
    print('Scraping ' + str(max_pages) + ' pages...')
    return max_pages
