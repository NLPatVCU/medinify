"""
Drug review scraper for Medinify.

This module scrapes comments from WebMD along with their rating.
Based on work by Amy Olex 11/13/17.
"""

import re
#import requests
from bs4 import BeautifulSoup

class WebMDScraper():
    """
    Class to scrap drug reviews from WebMD

    Attributes:
        all_pages: Boolean for whether or not to scrape all pages
        pages: int for # of pages to scrape if all_pages is 0
        review_list: List of review dictionary items
    """

    all_pages = True
    pages = 1
    review_list = []

    def __init__(self, all_pages=True, pages=1):
        self.all_pages = all_pages
        self.pages = pages

    def max_pages(self, input_url):
        """Finds number of review pages for this drug.

        Args:
            input_url: URL for the first page of reviews.
        Returns:
            (int) Highest page number
        """
        page = requests.get(input_url)
        soup = BeautifulSoup(page.text, 'html.parser')
        total_reviews_text = soup.find('span', {'class': 'totalreviews'}).text
        total_reviews = [int(s) for s in total_reviews_text.split() if s.isdigit()][0]
        max_pages = total_reviews // 5
        print('Found ' + str(total_reviews) + ' reviews.')
        print('Scraping ' + str(max_pages) + ' pages...')
        return max_pages

    def scrape_page(self, page_url):
        """Scrapes a single page for reviews and adds them to review_list

        Args:
            page_url: URL of the page to scrape.
        """
        page = requests.get(page_url)
        soup = BeautifulSoup(page.text, 'html.parser')
        reviews = soup.find_all('div', attrs={'class': 'userPost'})

        for review in reviews:
            comment = review.find('p', id=re.compile("^comFull*")).text
            comment = comment.replace('Comment:', '').replace('Hide Full Comment', '')
            comment = ' '.join(comment.splitlines())

            ratings = review.find_all('span', attrs={'class': 'current-rating'})
            effectiveness = int(ratings[0].text.replace('Current Rating:', '').strip())
            ease = int(ratings[1].text.replace('Current Rating:', '').strip())
            satisfaction = int(ratings[2].text.replace('Current Rating:', '').strip())

            self.review_list.append({'comment': comment,
                                     'effectiveness': effectiveness,
                                     'ease of use': ease,
                                     'satisfaction': satisfaction})

    def scrape(self, input_url):
        """Scrapes the reviews from WebMD

        Args:
            input_url : WebMD URL to scrape
        """

        print('Scraping WebMD...')

        self.review_list = []

        quote_page1 = input_url + '&pageIndex='
        quote_page2 = '&sortby=3&conditionFilter=-1'
        num_pages = 0

        if self.all_pages:
            num_pages = self.max_pages(input_url)
        else:
            num_pages = self.pages

        for i in range(num_pages):
            page_url = quote_page1 + str(i) + quote_page2
            self.scrape_page(page_url)

            page = i + 1
            if page % 10 == 0:
                print('Scraped ' + str(page) + ' pages...')

        print('Reviews scraped: ' + str(len(self.review_list)))

        return self.review_list
