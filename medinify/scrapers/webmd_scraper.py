"""
Drug review scraper for Medinify.

This module scrapes comments from WebMD along with their rating.
Based on work by Amy Olex 11/13/17.
"""

import re
import csv
import requests
from bs4 import BeautifulSoup

class WebMDScraper():
    """
    Class to scrap drug reviews from WebMD

    Attributes:
        output_path (str) : CSV file to output scraped information.
    """

    all_pages = True
    pages = 1
    review_list = []

    def __init__(self, all_pages=True, pages=1):
        self.all_pages = all_pages
        self.pages = pages

    def clean_comment(self, comment):
        """Cleans comment for proper CSV usage.
        Args:
            comment: Comment to be cleaned.
        Returns:
            The cleaned comment.
        """
        comment = comment.replace('Comment:', '').replace('Hide Full Comment', '')
        comment = ' '.join(comment.splitlines())
        return comment

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
            comment = self.clean_comment(comment)
            if comment:
                ratings = review.find_all('span', attrs={'class': 'current-rating'})
                calculated_rating = 0.0

                for rating in ratings:
                    calculated_rating += int(rating.text.replace('Current Rating:', '').strip())

                calculated_rating = calculated_rating / 3.0
                self.review_list.append({'comment': comment, 'rating': calculated_rating})

    def scrape(self, input_url, output_path):
        """Scrapes the reviews from WebMD

        Args:
            input_url : WebMD URL to scrape
            pages (int) : number of pages to scrape
        """

        self.review_list = []

        quote_page1 = input_url + '&pageIndex='
        quote_page2 = '&sortby=3&conditionFilter=-500'
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

        with open(output_path, 'w') as output_file:
            dict_writer = csv.DictWriter(output_file, ['comment', 'rating'])
            dict_writer.writeheader()
            dict_writer.writerows(self.review_list)

        print('Reviews scraped: ' + str(len(self.review_list)))
