"""Scrapes Drugs.com for drug reviews.
"""

import csv
import requests
from bs4 import BeautifulSoup

class DrugsScraper():
    """Scrapes Drugs.com for drug reviews.
    """

    def scrape(self, drug_url, pages=1):
        """Scrape for drug reviews.

        Args:
            drug_url: Drugs.com page to scrape
            output_path: Path to the file where the output should be sent
            pages (int): Number of pages to scrape
        """

        review_list = []

        for i in range(pages):
            url = drug_url + '?page=' + str(i+1)
            page = requests.get(url)
            soup = BeautifulSoup(page.text, 'html.parser')
            reviews = soup.find_all('div', {'class': 'user-comment'})

            for review in reviews:
                comment = review.find('span').text.lstrip('"').rstrip('"')
                rating = ''

                if review.find('div', {'class': 'rating-score'}):
                    rating = float(review.find('div', {'class': 'rating-score'}).text)

                review_list.append({'comment': comment, 'rating': rating})

        print('Reviews scraped: ' + str(len(review_list)))
        return review_list
