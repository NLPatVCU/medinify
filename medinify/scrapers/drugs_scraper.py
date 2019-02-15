"""Scrapes Drugs.com for drug reviews.
"""

import csv
import requests
from bs4 import BeautifulSoup

class DrugsScraper():
    """Scrapes Drugs.com for drug reviews.
    """

    all_pages = True
    pages = 1
    review_list = []

    def __init__(self, all_pages=True, pages=1):
        self.all_pages = all_pages
        self.pages = pages

    def max_pages(self, drug_url):
        """Finds number of review pages for this drug.

        Args:
            drug_url: URL for the first page of reviews.
        Returns:
            (int) Highest page number
        """
        page = requests.get(drug_url)
        soup = BeautifulSoup(page.text, 'html.parser')
        total_reviews_text = soup.find('span', {'class': 'totalreviews'}).text
        total_reviews = [int(s) for s in total_reviews_text.split() if s.isdigit()][0]
        max_pages = total_reviews // 5
        print('Found ' + str(total_reviews) + ' reviews.')
        print('Scraping ' + str(max_pages) + ' pages...')
        return max_pages

    def scrape(self, drug_url, output_path, pages=1):
        """Scrape for drug reviews.

        Args:
            drug_url: Drugs.com page to scrape
            output_path: Path to the file where the output should be sent
            pages (int): Number of pages to scrape
        """

        print('Scraping Drugs.com...')

        self.review_list = []
        
        if self.all_pages:
            num_pages = self.max_pages(drug_url)
        else:
            num_pages = pages

        for i in range(num_pages):
            url = drug_url + '?page=' + str(i+1)
            page = requests.get(url)
            soup = BeautifulSoup(page.text, 'html.parser')
            reviews = soup.find_all('div', {'class': 'user-comment'})

            for review in reviews:
                review_for = review.find('b').text
                comment = review.find('span').text.lstrip('"').rstrip('"')
                rating = ''

                if review.find('div', {'class': 'rating-score'}):
                    rating = float(review.find('div', {'class': 'rating-score'}).text)

                self.review_list.append({'comment': comment, 'for': review_for, 'rating': rating})

        with open(output_path, 'w') as output_file:
            dict_writer = csv.DictWriter(output_file, ['comment', 'for', 'rating'])
            dict_writer.writeheader()
            dict_writer.writerows(self.review_list)

        print('Reviews scraped: ' + str(len(self.review_list)))

    