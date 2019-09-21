"""
Scrapes Drugs.com for drug reviews.
"""

import requests
from bs4 import BeautifulSoup
from medinify.scrapers.scraper import Scraper
from tqdm import tqdm
import re


class DrugsScraper(Scraper):
    """Scrapes Drugs.com for drug reviews.
    """

    def scrape_page(self, url):
        """
        Scrapes a single page of drug reviews
        :param url: drug reviews page url
        :return:
        """
        assert url[:31] == 'https://www.drugs.com/comments/', 'Invalid Drugs.com Reviews Page URL'

        page = requests.get(url)
        soup = BeautifulSoup(page.text, 'html.parser')
        drug_name = re.split('User Reviews for | \(Page', soup.find('h1').text)[1]

        reviews = list(soup.find_all('div', {'class': 'ddc-comment'}))

        if len(reviews) == 0:
            print('No reviews found: %s' % url)
            return

        for review in reviews:
            row = {'comment': review.find('p', {'class': 'ddc-comment-content'}).find('span').text.replace('"', '')}
            rating = None
            if review.find('div', {'class', 'rating-score'}):
                rating = float(review.find('div', {'class', 'rating-score'}).text)
            row['rating'] = rating
            row['date'] = review.find('span', {'class': 'comment-date text-color-muted'}).text
            row['drug'] = drug_name.split()[0]
            if 'url' in self.data_collected:
                row['url'] = url
            if 'user id' in self.data_collected:
                id_ = None
                if review.find('span', {'class', 'user-name user-type user-type-1_standard_member'}):
                    id_ = review.find('span', {'class', 'user-name user-type user-type-1_standard_member'}).text
                elif review.find('span', {'class': 'user-name user-type user-type-2_non_member'}):
                    id_ = review.find('span', {'class': 'user-name user-type user-type-2_non_member'}).text
                row['user id'] = id_
            self.reviews.append(row)

    def scrape(self, url):
        """
        Scrapes all reviews of a given drug
        :param url: drug reviews url
        """
        super().scrape(url)
        front_page = requests.get(url)
        front_page_soup = BeautifulSoup(front_page.text, 'html.parser')
        title = front_page_soup.find('h1').text

        if 'User Reviews for ' in title:
            drug_name = re.split('User Reviews for | \(Page', front_page_soup.find('h1').text)[1]
        else:
            print('Invalid URL entered: %s' % url)
            return

        print('Scraping Drugs.com for %s Reviews...' % drug_name)
        base_url = url + '?page='

        num_pages = max_pages(url)
        for i in tqdm(range(num_pages)):
            full_url = base_url + str(i + 1)
            self.scrape_page(full_url)

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

        characters = list('+'.join(drug_name.lower().split()))
        name = ''.join([x if x.isalnum() or x == '+' else hex(ord(x)).replace('0x', '%') for x in characters])

        search_url = 'https://www.drugs.com/search.php?searchterm=' + name
        search_page = requests.get(search_url)
        search_soup = BeautifulSoup(search_page.text, 'html.parser')

        reviews_url = None

        if search_soup.find('p', {'class': 'user-reviews-title mgb-1'}):
            reviews_url = 'https://www.drugs.com' + search_soup.find(
                'p', {'class': 'user-reviews-title mgb-1'}).find('a').attrs['href']
        elif search_soup.find('img', {'src': '/img/icons/star.png'}):
            url = search_soup.find('h3').find('a').attrs['href']
            candidate_url = url[:22] + 'comments' + url[21:]
            reviews_page = requests.get(candidate_url)
            reviews_soup = BeautifulSoup(reviews_page.text, 'html.parser')
            if not reviews_soup.find('h1').text[:16] == 'User Reviews for':
                reviews_url = candidate_url

        if return_multiple and reviews_url:
            print('Found 1 Review Page for %s' % drug_name)
            return [reviews_url]
        elif reviews_url:
            return reviews_url
        else:
            print('Found no %s reviews' % drug_name)
            return None


def max_pages(drug_url):
    """Finds number of review pages for this drug.

    Args:
        drug_url: URL for the first page of reviews.
    Returns:
        (int) Highest page number
    """
    page = requests.get(drug_url)
    soup = BeautifulSoup(page.text, 'html.parser')
    table_footer = None
    if soup.find('table', {'class': 'data-list ddc-table-sortable'}):
        if soup.find('table', {'class': 'data-list ddc-table-sortable'}).find('tfoot'):
            table_footer = soup.find('table', {
                'class': 'data-list ddc-table-sortable'}).find('tfoot').find('tr').find_all('th')
    if not table_footer:
        return 0
    total_reviews = int(''.join([ch for ch in table_footer[2].text if ch.isdigit()]))

    max_pages_ = total_reviews // 25
    if total_reviews % 25 != 0:
        max_pages_ += 1

    print('Found %d reviews (%d pages).' % (total_reviews, max_pages_))
    return max_pages_

