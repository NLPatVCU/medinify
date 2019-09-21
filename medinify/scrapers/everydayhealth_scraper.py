"""
EverydayHealth.com drug review scraper
"""

import re
import requests
from bs4 import BeautifulSoup
from medinify.scrapers.scraper import Scraper
from tqdm import tqdm


class EverydayHealthScraper(Scraper):
    """Scrapes EverydayHealth.com for drug reviews.
    """

    def __init__(self, collect_user_ids=False, collect_urls=False):
        super(EverydayHealthScraper, self).__init__(collect_user_ids, collect_urls)
        if collect_user_ids:
            raise AttributeError('EverydayHealth.com does not collect user id data')

    def scrape_page(self, url):
        """
        Scrapes a single page of drug reviews
        :param url: drug reviews page url
        """
        assert url[:30] == 'https://www.everydayhealth.com', \
            'Url must be link to an EverydayHealth.com reviews page'

        page = requests.get(url)
        soup = BeautifulSoup(page.text, 'html.parser')
        reviews = soup.find_all('div', {'itemprop': 'review'})
        drug_name = soup.find('title').text.split()[0]

        if len(reviews) == 0:
            print('No reviews found for drug %s' % drug_name)
            return

        for review in reviews:
            row = {'comment': review.find('p', {'itemprop': 'reviewBody'}).text[:-7]}
            if type(row['comment']) == float:
                print('Skipping invalid comment (Not a string)')
                continue
            row['rating'] = None
            if review.find('span', {'itemprop': 'reviewRating'}):
                row['rating'] = float(review.find('span', {'itemprop': 'reviewRating'}).text)
            row['date'] = review.find('span', {'class': 'time'}).attrs['content']
            row['drug'] = drug_name
            if 'url' in self.data_collected:
                row['url'] = url
            self.reviews.append(row)

    def scrape(self, url):
        """
        Scrapes all reviews of a given drug
        :param url: drug reviews url
        """
        super().scrape(url)
        front_page = requests.get(url)
        front_page_soup = BeautifulSoup(front_page.text, 'html.parser')
        if front_page_soup.find('span', {'itemprop': 'name'}):
            drug_name = front_page_soup.find('span', {'itemprop': 'name'}).text
        else:
            print('Invalid URL entered: %s' % url)
            return
        print('Scraping EverydayHealth for %s Reviews...' % drug_name)

        num_pages = max_pages(url)

        for i in tqdm(range(num_pages)):
            page_url = url + '/' + str(i + 1)
            self.scrape_page(page_url)

    def get_url(self, drug_name, return_multiple=False):
        """
        Given a drug name, finds the drug review page(s) on a given review forum
        :param drug_name: name of drug being searched for
        :param return_multiple: if multiple urls are found, whether or not to return all of them
        :return: drug url on given review forum
        """
        if not drug_name or len(drug_name) < 4:
            print('%s name too short; Please manually search for such reviews' % drug_name)
            return []

        review_urls = []
        drug = re.sub('\s+', '-', drug_name.lower())
        search_url = 'https://www.everydayhealth.com/drugs/' + drug + '/reviews'
        page = requests.get(search_url)
        search_soup = BeautifulSoup(page.text, 'html.parser')
        if 'Reviews' in search_soup.find('title').text.split():
            review_urls.append(search_url)
        if return_multiple and len(review_urls) > 0:
            print('Found %d Review Page(s) for %s' % (len(review_urls), drug_name))
            return review_urls
        elif len(review_urls) > 0:
            return review_urls[0]
        else:
            print('Found no %s reviews' % drug_name)
            return None


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
    pages = int(max_pages_foot[2])

    print('Found %d reviews (%d pages).' % (total_reviews, pages))
    return pages
