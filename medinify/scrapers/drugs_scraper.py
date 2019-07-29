"""
Scrapes Drugs.com for drug reviews.
"""

import pandas as pd
import requests
from bs4 import BeautifulSoup
from medinify.scrapers.scraper import Scraper
import warnings


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
        drug_name = soup.find('h1').text.replace('User Reviews for ', '')
        reviews = soup.find_all('div', {'class': 'ddc-comment'})

        if len(reviews) == 0:
            warnings.warn('No reviews found for drug {}'.format(drug_name), UserWarning)
            return 1

        rows = {'comment': []}
        if 'rating' in self.data_collected:
            rows['rating'] = []
        if 'date' in self.data_collected:
            rows['date'] = []
        if 'drug' in self.data_collected:
            rows['drug'] = []
        if 'user id' in self.data_collected:
            rows['user id'] = []
        if 'url' in self.data_collected:
            rows['url'] = []

        for review in reviews:
            comment = review.find('p', {'class': 'ddc-comment-content'}).find('span').text.replace('"', '')
            rows['comment'].append(comment)
            if 'rating' in self.data_collected:
                rating = None
                if review.find('div', {'class', 'rating-score'}):
                    rating = float(review.find('div', {'class', 'rating-score'}).text)
                rows['rating'].append(rating)
            if 'date' in self.data_collected:
                rows['date'].append(review.find('span', {'class': 'comment-date text-color-muted'}).text)
            if 'drug' in self.data_collected:
                rows['drug'] = drug_name.split()[0]
            if 'url' in self.data_collected:
                rows['url'].append(url)
            if 'user id' in self.data_collected:
                id_ = None
                if review.find('span', {'class', 'user-name user-type user-type-1_standard_member'}):
                    id_ = review.find('span', {'class', 'user-name user-type user-type-1_standard_member'}).text
                elif review.find('span', {'class': 'user-name user-type user-type-2_non_member'}):
                    id_ = review.find('span', {'class': 'user-name user-type user-type-2_non_member'}).text
                rows['user id'].append(id_)

        scraped_data = pd.DataFrame(rows, columns=self.data_collected)
        self.dataset = self.dataset.append(scraped_data, ignore_index=True)

    def scrape(self, url):
        """
        Scrapes all reviews of a given drug
        :param url: drug reviews url
        """
        print('Scraping Drugs.com...')

        base_url = url + '?page='
        num_pages = max_pages(url)

        for i in range(num_pages):
            full_url = base_url + str(i + 1)
            self.scrape_page(full_url)

            if (i + 1) % 10 == 0:
                print('Scraped {} of {} pages...'.format(i + 1, num_pages))

    def get_url(self, drug_name):
        """
        Given a drug name, finds the drug review page(s) on a given review forum
        :param drug_name: name of drug being searched for
        :return: drug url on given review forum
        """
        if not drug_name or len(drug_name) < 4:
            print('{} name too short; Please manually search for such reviews'.format(drug_name))
            return []

        characters = list('+'.join(drug_name.lower().split()))
        name = ''.join([x if x.isalnum() or x == '+' else hex(ord(x)).replace('0x', '%') for x in characters])

        search_url = 'https://www.drugs.com/search.php?searchterm=' + name
        search_page = requests.get(search_url)
        search_soup = BeautifulSoup(search_page.text, 'html.parser')

        if search_soup.find('p', {'class': 'user-reviews-title mgb-1'}):
            reviews_url = 'https://www.drugs.com' + search_soup.find(
                'p', {'class': 'user-reviews-title mgb-1'}).find('a').attrs['href']
            return [reviews_url]
        elif search_soup.find('img', {'src': '/img/icons/star.png'}):
            url = search_soup.find('h3').find('a').attrs['href']
            reviews_url = url[:22] + 'comments' + url[21:]
            reviews_page = requests.get(reviews_url)
            reviews_soup = BeautifulSoup(reviews_page.text, 'html.parser')
            if reviews_soup.find('h1').text[:16] == 'User Reviews for':
                return [reviews_url]

        return []


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

    print('Found ' + str(total_reviews) + ' reviews.')
    print('Scraping ' + str(max_pages_) + ' pages...')
    return max_pages_

