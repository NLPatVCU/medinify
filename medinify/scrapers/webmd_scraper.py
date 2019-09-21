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
from tqdm import tqdm
import string


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
            print('No reviews found for drug %s' % drug_name)
            return

        for review in reviews:
            row = {}
            comment = review.find('p', {'id': re.compile("^comFull*")}).text
            if type(comment) == float:
                print('Skipping invalid comment (Not a string)')
                continue
            comment = re.sub('Comment:|Hide Full Comment', '', comment)
            row['comment'] = comment

            rating_dict = {}
            rates = review.find_all('span', attrs={'class': 'current-rating'})
            rating_dict['effectiveness'] = float(rates[0].text.replace('Current Rating:', '').strip())
            rating_dict['ease of use'] = float(rates[1].text.replace('Current Rating:', '').strip())
            rating_dict['satisfaction'] = float(rates[2].text.replace('Current Rating:', '').strip())
            row['rating'] = rating_dict
            row['date'] = review.find('div', {'class': 'date'}).text
            row['drug'] = drug_name

            if 'url' in self.data_collected:
                row['url'] = url
            if 'user id' in self.data_collected:
                row['user id'] = review.find('p', {'class': 'reviewerInfo'}).text.replace('Reviewer: ', '')
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
        if 'User Reviews & Ratings - ' in title:
            drug_name = re.sub('User Reviews & Ratings - ', '', title)
            drug_name = string.capwords(drug_name)
        else:
            print('Invalid URL entered: %s' % url)
            return
        print('Scraping WebMD for %s Reviews...' % drug_name)

        quote_page1 = url + '&pageIndex='
        quote_page2 = '&sortby=3&conditionFilter=-1'

        num_pages = max_pages(url)

        for i in tqdm(range(num_pages)):
            page_url = quote_page1 + str(i) + quote_page2
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
        try:
            page = requests.get(input_url)
            soup = BeautifulSoup(page.text, 'html.parser')
            if 'Be the first to share your experience with this treatment.' in \
                    soup.find('div', {'id': 'heading'}).text:
                return 0
            break
        except AttributeError:
            print('Ran into AttributeError. Waiting 10 seconds and retrying...')
            sleep(10)

    total_reviews_text = soup.find('span', {'class': 'totalreviews'}).text
    total_reviews = [int(s) for s in total_reviews_text.split() if s.isdigit()][0]

    pages = total_reviews // 5
    if total_reviews % 5 != 0:
        pages += 1

    print('Found %d reviews (%d pages).' % (total_reviews, pages))
    return pages
