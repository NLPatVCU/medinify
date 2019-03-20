"""
Drug review scraper for Iodine.com
"""

import re
import csv
import requests
from bs4 import BeautifulSoup


class IodineScraper():
    """Objective of script is to scrape iodine.com for drug reviews """

    def scraper(self, drug_url):
        """
        Args: drug_url: iodine.com page
        output_path: file path for output
        pages (int): number of pages to scrape from arg """

        #initialize
        review_list = []
        #types of ratings (best -> worst)
        worth_it = 0
        worked_well = 0
        big_hassle = 0

        #iter through multi-pages
        page = requests.get(drug_url)
        soup = BeautifulSoup(page.text, 'html.parser')
        reviews = soup.find_all(
            'div', {'class': 'tri bg-white relative depth-1 p-3 col mo-m-t-2'})
        for review in reviews:
            if len(review) >= 2:
                if review.find('div', {'class': "purple"}):
                    worth_it += 1
                elif review.find('div', {'class': "navy-blue"}):
                    worked_well += 1
                elif review.find('div', {'class': "red-3"}):
                    big_hassle += 1
                
                comment = review.find('span', {'class': None})
                comment = re.sub('<[/]span>', '', comment.decode())
                comment = re.sub('<span>', '', comment)

                review_list.append({
                    'comment': comment,
                    'worth it': worth_it,
                    'worked well': worked_well,
                    'big hassle': big_hassle
                })

        print('Reviews scraped: ' + str(len(review_list)))
        return review_list
