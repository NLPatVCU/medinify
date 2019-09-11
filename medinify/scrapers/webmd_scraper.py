"""
Drug review scraper for Medinify.
This module scrapes comments from WebMD along with their rating.
Based on work by Amy Olex 11/13/17.
"""

import re
from time import sleep
import requests
from bs4 import BeautifulSoup
from medinify.scrapers.scraper import Scraper
import pandas as pd
import warnings
from tqdm import tqdm


class WebMDScraper(Scraper):
    """
    Class to scrap drug reviews from WebMD
    """

    def scrape_page(self, url):
        """
        Scrapes a single page of reviews
        :param url: String, url for review page
        """
        assert url[:39] == 'https://www.webmd.com/drugs/drugreview-', 'Url must be link to a WebMD reviews page'

        page = requests.get(url)
        soup = BeautifulSoup(page.text, 'html.parser')
        drug_name = soup.find('h1').text.replace('User Reviews & Ratings - ', '')
        reviews = soup.find_all('div', attrs={'class': 'userPost'})

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
            comment = review.find('p', {'id': re.compile("^comFull*")}).text
            if type(comment) == float:
                continue
            clean_comment = re.sub('Comment:|Hide Full Comment', '', comment)
            rows['comment'].append(clean_comment)

            rating_set = {}
            rates = review.find_all('span', attrs={'class': 'current-rating'})
            rating_set['effectiveness'] = float(rates[0].text.replace('Current Rating:', '').strip())
            rating_set['ease of use'] = float(rates[1].text.replace('Current Rating:', '').strip())
            rating_set['satisfaction'] = float(rates[2].text.replace('Current Rating:', '').strip())
            rows['rating'].append(rating_set)
            rows['date'].append(review.find('div', {'class': 'date'}).text)
            rows['drug'].append(drug_name)

            if 'url' in self.data_collected:
                rows['url'].append(url)
            if 'user id' in self.data_collected:
                rows['user id'].append(review.find('p', {'class': 'reviewerInfo'}).text.replace('Reviewer: ', ''))

        scraped_data = pd.DataFrame(rows, columns=self.data_collected)
        self.dataset = self.dataset.append(scraped_data, ignore_index=True)

    def scrape(self, url):
        """
        Scrapes all reviews of a given drug
        :param url: drug reviews url
        """
        if self.dataset.shape[0] > 0:
            print('Clearing scraper\'s pre-existent dataset of {} '
                  'collected reviews...'.format(self.dataset.shape[0]))
            self.dataset = pd.DataFrame(columns=self.data_collected)
        print('Scraping WebMD...')

        quote_page1 = url + '&pageIndex='
        quote_page2 = '&sortby=3&conditionFilter=-1'

        pages = max_pages(url)

        for i in tqdm(range(pages)):
            page_url = quote_page1 + str(i) + quote_page2
            self.scrape_page(page_url)

    def get_url(self, drug_name):
        """
        Given a drug name, finds the drug review page(s) on a given review forum
        :param drug_name: name of drug being searched for
        :return: drug url on given review forum
        """

        if not drug_name or len(drug_name) < 4:
            print('{} name too short; Please manually search for such reviews'.format(drug_name))
            return []

        characters = list(drug_name.lower())
        name = ''.join([x if x.isalnum() else hex(ord(x)).replace('0x', '%') for x in characters])

        url = 'https://www.webmd.com/drugs/2/search?type=drugs&query=' + name
        page = requests.get(url)
        search_soup = BeautifulSoup(page.text, 'html.parser')

        review_urls = []

        if search_soup.find('a', {'class': 'drug-review'}):
            review_url = 'https://www.webmd.com' + search_soup.find('a', {'class': 'drug-review'}).attrs['href']
            review_urls.append(review_url)

        elif search_soup.find('ul', {'class': 'exact-match'}):
            exact_matches = search_soup.find('ul', {'class': 'exact-match'})
            search_links = ['https://www.webmd.com' + x.attrs['href'] for x in exact_matches.find_all('a')]
            for info_page in search_links:
                info = requests.get(info_page)
                info_soup = BeautifulSoup(info.text, 'html.parser')
                review_url = 'https://www.webmd.com' + info_soup.find('a', {'class': 'drug-review'}).attrs['href']
                review_urls.append(review_url)

        print('Found {} Review Page(s) for {}'.format(len(review_urls), drug_name))
        return review_urls


def max_pages(input_url):
    """Finds number of review pages for this drug.
    Args:
        input_url: URL for the first page of reviews.
    Returns:
        (int) Highest page number
    """
    while True:
        try:
            page = requests.get(input_url)
            soup = BeautifulSoup(page.text, 'html.parser')
            if 'Be the first to share your experience with this treatment.' in soup.find('div', {'id': 'heading'}).text:
                return 0
            break
        except AttributeError:
            print('Ran into AttributeError. Waiting 10 seconds and retrying...')
            sleep(10)

    total_reviews_text = soup.find('span', {'class': 'totalreviews'}).text
    total_reviews = [int(s) for s in total_reviews_text.split() if s.isdigit()][0]

    # Does the equivalent of max_pages = ceil(total_reviews / 5) without the math library
    max_pages = total_reviews // 5
    if total_reviews % 5 != 0:
        max_pages += 1

    print('Found ' + str(total_reviews) + ' reviews.')
    print('Scraping ' + str(max_pages) + ' pages...')
    return max_pages
