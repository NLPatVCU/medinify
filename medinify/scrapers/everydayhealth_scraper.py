"""
Drug review scraper for EverydayHealth.com
Implements the ability to collect the following data:

    -> Comments (Review text)
    -> 5-Point Scale Star Ratings
    -> Post Dates
    -> Each review's associated url
    -> Each review's associated drug name

and to search for review urls given a drug name
"""
import re
import requests
from bs4 import BeautifulSoup
from medinify.scrapers.scraper import Scraper
from tqdm import tqdm


class EverydayHealthScraper(Scraper):
    """
    The EverydayHealthScraper class implements drug review scraping functionality for EverydayHealth.com

    Attributes:
        collect_urls:    (Boolean) Whether or not to collect each review's associated url
        reviews:         (list[dict]) Scraped review data
    """

    nickname = 'everydayhealth'

    def scrape_page(self, url):
        """
        Collects data from one page of EverydayHealth drug reviews into the 'reviews' attribute,
        a list of dictionaries containing each requisite piece of data
        (comment, rating, date, drug, and url (if specified))
        :param url: (str) the url for the page to be scraped
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
            if self.collect_urls:
                row['url'] = url
            self.reviews.append(row)

    def scrape(self, url):
        """
        Scrapes all review pages for a given drug on EverydayHealth into 'reviews' attribute
        :param url: (str) url to the first page of drug reviews for this drug
        """
        super().scrape(url)
        front_page = requests.get(url)
        front_page_soup = BeautifulSoup(front_page.text, 'html.parser')

        try:
            drug_name = front_page_soup.find('span', {'itemprop': 'name'}).text
        except AttributeError:
            print('Invalid URL entered: %s' % url)
            return 0

        print('Scraping EverydayHealth for %s Reviews...' % drug_name)
        num_pages = max_pages(url)

        for i in tqdm(range(num_pages)):
            page_url = url + '/' + str(i + 1)
            self.scrape_page(page_url)

    def get_url(self, drug_name):
        """
        Searches EverydayHealth for reviews for a certain drug
        :param drug_name: (str) name of the drug to search for
        :return review_url: (str or None) if reviews for a drug with a matching name are found,
            this is the url for the first page of those reviews
            if a match was not found, returns None
        """
        if len(drug_name) < 4:
            print('%s name too short; Please manually search for such reviews' % drug_name)
            return None
        drug = re.sub('\s+', '-', drug_name.lower())
        search_url = 'https://www.everydayhealth.com/drugs/' + drug + '/reviews'
        page = requests.get(search_url)
        search_soup = BeautifulSoup(page.text, 'html.parser')

        review_url = None
        if 'Reviews' in search_soup.find('title').text.split():
            review_url = search_url

        return review_url


def max_pages(input_url):
    """
    Get the number of review pages for a given drug
    :param input_url: (str) first page of reviews for a drug
    :return pages: (int) number of review pages on EverydayHealth.com for the drug
    """
    page = requests.get(input_url)
    soup = BeautifulSoup(page.text, 'html.parser')

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
