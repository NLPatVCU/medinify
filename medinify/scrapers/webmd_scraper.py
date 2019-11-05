"""
Drug review scraper for WebMD.com
Implements the ability to collect the following data:

    -> Comments (Review text)
    -> 5-Point Scale Star Ratings ('Effectiveness', 'Ease of Use', and 'Satisfaction')
    -> Post Dates
    -> Use IDs
    -> Each review's associated url
    -> Each review's associated drug name

and to search for review urls given a drug name
"""
import re
import requests
from bs4 import BeautifulSoup
from medinify.scrapers.scraper import Scraper
from tqdm import tqdm
import string


class WebMDScraper(Scraper):
    """
    The WebMDScraper class implements drug review scraping functionality for WebMD.com

    Attributes:
        collect_urls:    (Boolean) Whether or not to collect each review's associated url
        collect_use_ids: (Boolean) Whether or not to collect user ids
        reviews:         (list[dict]) Scraped review data
    """

    nickname = 'webmd'

    def __init__(self, collect_user_ids=False, collect_urls=False):
        """
        Constructor for WebMD scraper, used to collecting review data from WebMD.com
        Sets up what data ought to be collected, and sets up how that data will be
        stored (in the list attribute 'reviews')
        :param collect_user_ids: (Boolean) whether or not this scraper will collect user ids
        :param collect_urls: (Boolean) whether or not this scraper will collect each drug review's associated url
        """
        super().__init__(collect_urls=collect_urls)
        self.collect_user_ids = collect_user_ids

    def scrape_page(self, url):
        """
        Collects data from one page of WebMD drug reviews into the 'reviews' attribute,
        a list of dictionaries containing each requisite piece of data
        (comment, rating, date, drug, user id (if specified), and url (if specified))
        :param url: (str) the url for the page to be scraped
        """
        assert url[:39] == 'https://www.webmd.com/drugs/drugreview-', 'Url must be link to a WebMD reviews page'

        page = requests.get(url)
        soup = BeautifulSoup(page.text, 'html.parser')
        drug_name = soup.find('h1').text.replace('User Reviews & Ratings - ', '')
        reviews = soup.find_all('div', attrs={'class': 'userPost'})

        if len(reviews) == 0:
            print('No reviews found for drug %s' % drug_name)
            return 0

        for review in reviews:
            row = {}
            comment = review.find('p', {'id': re.compile("^comFull*")}).text
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

            if self.collect_urls:
                row['url'] = url
            if self.collect_user_ids:
                row['user id'] = review.find('p', {'class': 'reviewerInfo'}).text.replace('Reviewer: ', '').strip()
            self.reviews.append(row)

    def scrape(self, url):
        """
        Scrapes all review pages for a given drug on WebMD into 'reviews' attribute
        :param url: (str) url to the first page of drug reviews for this drug
        """
        super().scrape(url)
        front_page = requests.get(url)
        front_page_soup = BeautifulSoup(front_page.text, 'html.parser')
        try:
            title = front_page_soup.find('h1').text
            assert 'User Reviews & Ratings - ' in title
            drug_name = re.sub('User Reviews & Ratings - ', '', title)
            drug_name = string.capwords(drug_name)
        except AssertionError:
            print('Invalid URL entered: %s' % url)
            return 0
        except AttributeError:
            print('Invalid URL entered: %s' % url)
            return 0

        print('Scraping WebMD for %s Reviews...' % drug_name)

        quote_page1 = url + '&pageIndex='
        quote_page2 = '&sortby=3&conditionFilter=-1'

        num_pages = max_pages(url)

        for i in tqdm(range(num_pages)):
            page_url = quote_page1 + str(i) + quote_page2
            self.scrape_page(page_url)

    def get_url(self, drug_name):
        """
        Searches WebMD for reviews for a certain drug
        :param drug_name: (str) name of the drug to search for
        :return review_url: (str or None) if reviews for a drug with a matching name are found,
            this is the url for the first page of those reviews
            if a match was not found, returns None
        """
        if len(drug_name) < 4:
            print('%s name too short; Please manually search for such reviews' % drug_name)
            return None
        characters = list(drug_name.lower())
        name = ''.join([x if x.isalnum() else hex(ord(x)).replace('0x', '%') for x in characters])

        search_url = 'https://www.webmd.com/drugs/2/search?type=drugs&query=' + name
        search_page = requests.get(search_url)
        search_soup = BeautifulSoup(search_page.text, 'html.parser')

        review_url = None

        if search_soup.find('a', {'class': 'drug-review'}):
            review_url = 'https://www.webmd.com' + search_soup.find('a', {'class': 'drug-review'}).attrs['href']

        elif search_soup.find('ul', {'class': 'exact-match'}):
            exact_matches = search_soup.find('ul', {'class': 'exact-match'})
            search_link = 'https://www.webmd.com' + exact_matches.find('a').attrs['href']
            info = requests.get(search_link)
            info_soup = BeautifulSoup(info.text, 'html.parser')
            review_url = 'https://www.webmd.com' + info_soup.find('a', {'class': 'drug-review'}).attrs['href']

        return review_url


def max_pages(input_url):
    """
    Get the number of review pages for a given drug
    :param input_url: (str) first page of reviews for a drug
    :return pages: (int) number of review pages on WebMD for the drug
    """
    page = requests.get(input_url)
    soup = BeautifulSoup(page.text, 'html.parser')
    if 'Be the first to share your experience with this treatment.' in \
            soup.find('div', {'id': 'heading'}).text:
        return 0

    total_reviews_text = soup.find('span', {'class': 'totalreviews'}).text
    total_reviews = [int(s) for s in total_reviews_text.split() if s.isdigit()][0]
    pages = total_reviews // 5
    if total_reviews % 5 != 0:
        pages += 1

    print('Found %d reviews (%d pages).' % (total_reviews, pages))
    return pages
