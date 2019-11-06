"""
Drug review scraper for Drugs.com
Implements the ability to collect the following data:

    -> Comments (Review text)
    -> Star Rating (0.0-10.0 scale)
    -> Post Dates
    -> Use IDs
    -> Each review's associated url
    -> Each review's associated drug name

and to search for review urls given a drug name
"""
import requests
from bs4 import BeautifulSoup
from medinify.scrapers.scraper import Scraper
from tqdm import tqdm
import re


class DrugsScraper(Scraper):
    """
    The DrugsScraper class implements drug review scraping functionality for Drugs.com

    Attributes:
        collect_urls:    (Boolean) Whether or not to collect each review's associated url
        collect_use_ids: (Boolean) Whether or not to collect user ids
        reviews:         (list[dict]) Scraped review data
    """

    nickname = 'drug'

    def __init__(self, collect_user_ids=False, collect_urls=False):
        """
        Constructor for Drugs scraper, used to collecting review data from Drugs.com
        Sets up what data ought to be collected, and sets up how that data will be
        stored (in the list attribute 'reviews')
        :param collect_user_ids: (Boolean) whether or not this scraper will collect user ids
        :param collect_urls: (Boolean) whether or not this scraper will collect each drug review's associated url
        """
        super().__init__(collect_urls=collect_urls)
        self.collect_user_ids = collect_user_ids

    def scrape_page(self, url):
        """
        Collects data from one page of Drugs.com drug reviews into the 'reviews' attribute,
        a list of dictionaries containing each requisite piece of data
        (comment, rating, date, drug, user id (if specified), and url (if specified))
        :param url: (str) the url for the page to be scraped
        """
        assert url[:31] == 'https://www.drugs.com/comments/', 'Invalid Drugs.com Reviews Page URL'

        page = requests.get(url)
        soup = BeautifulSoup(page.text, 'html.parser')
        drug_name = re.split('User Reviews for | \(Page', soup.find('h1').text)[1]

        reviews = list(soup.find_all('div', {'class': 'ddc-comment'}))

        if len(reviews) == 0:
            print('No reviews found: %s' % url)
            return 0

        for review in reviews:
            row = {'comment': review.find('p', {'class': 'ddc-comment-content'}).find('span').text.replace('"', '')}
            rating = None
            if review.find('div', {'class', 'rating-score'}):
                rating = float(review.find('div', {'class', 'rating-score'}).text)
            row['rating'] = rating
            row['date'] = review.find('span', {'class': 'comment-date text-color-muted'}).text
            row['drug'] = drug_name.split()[0]
            if self.collect_urls:
                row['url'] = url
            if self.collect_user_ids:
                id_ = None
                if review.find('span', {'class', 'user-name user-type user-type-1_standard_member'}):
                    id_ = review.find('span', {'class', 'user-name user-type user-type-1_standard_member'}).text
                elif review.find('span', {'class': 'user-name user-type user-type-2_non_member'}):
                    id_ = review.find('span', {'class': 'user-name user-type user-type-2_non_member'}).text
                row['user id'] = id_
            self.reviews.append(row)

    def scrape(self, url):
        """
        Scrapes all review pages for a given drug on Drugs.com into 'reviews' attribute
        :param url: (str) url to the first page of drug reviews for this drug
        """
        super().scrape(url)
        front_page = requests.get(url)
        front_page_soup = BeautifulSoup(front_page.text, 'html.parser')
        try:
            title = front_page_soup.find('h1').text
            assert 'User Reviews for ' in title
            drug_name = re.split('User Reviews for | \(Page', front_page_soup.find('h1').text)[1]
        except AssertionError:
            print('Invalid URL entered: %s' % url)
            return 0
        except AttributeError:
            print('Invalid URL entered: %s' % url)
            return 0

        print('Scraping Drugs.com for %s Reviews...' % drug_name)
        base_url = url + '?page='

        num_pages = max_pages(url)
        for i in tqdm(range(num_pages)):
            full_url = base_url + str(i + 1)
            self.scrape_page(full_url)

    def get_url(self, drug_name):
        """
        Searches Drugs.com for reviews for a certain drug
        :param drug_name: (str) name of the drug to search for
        :return review_url: (str or None) if reviews for a drug with a matching name are found,
            this is the url for the first page of those reviews
            if a match was not found, returns None
        """
        if len(drug_name) < 4:
            print('%s name too short; Please manually search for such reviews' % drug_name)
            return None
        characters = list('+'.join(drug_name.lower().split()))
        name = ''.join([x if x.isalnum() or x == '+' else hex(ord(x)).replace('0x', '%') for x in characters])

        search_url = 'https://www.drugs.com/search.php?searchterm=' + name
        search_page = requests.get(search_url)
        search_soup = BeautifulSoup(search_page.text, 'html.parser')

        reviews_url = None

        if search_soup.find('p', {'class': 'user-reviews-title mgb-1'}):
            reviews_url = 'https://www.drugs.com' + search_soup.find(
                'p', {'class': 'user-reviews-title mgb-1'}).find('a').attrs['href']

        return reviews_url


def max_pages(input_url):
    """
    Get the number of review pages for a given drug
    :param input_url: (str) first page of reviews for a drug
    :return pages: (int) number of review pages on Drugs.com for the drug
    """
    page = requests.get(input_url)
    soup = BeautifulSoup(page.text, 'html.parser')
    table_footer = None
    if soup.find('table', {'class': 'data-list ddc-table-sortable'}):
        if soup.find('table', {'class': 'data-list ddc-table-sortable'}).find('tfoot'):
            table_footer = soup.find('table', {
                'class': 'data-list ddc-table-sortable'}).find('tfoot').find('tr').find_all('th')
    if not table_footer:
        return 0
    total_reviews = int(''.join([ch for ch in table_footer[2].text if ch.isdigit()]))

    pages = total_reviews // 25
    if total_reviews % 25 != 0:
        pages += 1

    print('Found %d reviews (%d pages).' % (total_reviews, pages))
    return pages

