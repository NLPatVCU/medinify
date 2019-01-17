"""Scrapes RXList for drug reviews.
"""

import re
import csv
import argparse
import requests
from bs4 import BeautifulSoup

class RXListScraper():
    """Scrapes RXList for drug reviews.
    """

    def scrape(self, drug_url, output_path):
        """Scrape for drug reviews.
        """
        review_list = []
        page = requests.get(drug_url)
        soup = BeautifulSoup(page.text, 'html.parser')

        reviews = soup.find_all('div', {'class': 'commentList'})

        for review in reviews:
            comment_from = review.find('span', {'class': 'commentFrom'}).text
            comment_from = comment_from.replace('Comment from: ', '')

            comment_date = review.find('span', {'class': 'commentDate'}).text
            comment_date = comment_date.replace('Published: ', '')

            comment = review.find('div', {'class': 'patComment'}).text
            comment = comment.lstrip("\n\r").rstrip("\n\r")

            review_list.append({'published': comment_date, 'from': comment_from, 'comment': comment})

        with open(output_path, 'w') as output_file:
            writer = csv.DictWriter(output_file, ['published', 'from', 'comment'])
            writer.writeheader()
            writer.writerows(review_list)

        print('Reviews scraped: ' + str(len(review_list)))
