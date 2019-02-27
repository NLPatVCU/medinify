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
        # document.querySelector('#content > div.contentBox > div.responsive-table-wrap-mobile > table > tfoot > tr > th:nth-child(3)')
        # //*[@id="content"]/div[2]/div[2]/table/tfoot/tr/th[3]
        # 
        table_footer = soup.find('table', {'class': 'data-list ddc-table-sortable'}).find('tfoot').find('tr').find_all('th')
        total_reviews = int(table_footer[2].get_text().split()[0])

        max_pages = total_reviews // 25
        print('Found ' + str(total_reviews) + ' reviews.')
        print('Scraping ' + str(max_pages) + ' pages...')
        return max_pages

    def scrape(self, drug_url, pages=1):
        """Scrape for drug reviews.

        Args:
            drug_url: Drugs.com page to scrape
            
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

        print('Reviews scraped: ' + str(len(self.review_list)))

    