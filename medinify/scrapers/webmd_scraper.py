"""
Drug review scraper for Medinify.

This module scrapes comments from WebMD along with their rating. 
Based on work by Amy Olex 11/13/17.
"""

import re
import time
import csv
import argparse
import requests
from bs4 import BeautifulSoup

class WebMDScraper():
    """
    Class to scrap drug reviews from WebMD

    Attributes:
        output_path (str) : CSV file to output scraped information        
    """

    output_path = ""

    def __init__(self, output_path):
        self.output_path = output_path


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


    def scrape(self, input_url, pages):
        """ 
        Scraps the reviews from WebMD

        Args:
            input_url : WebMD URL to scrap
            pages (int) : number of pages to scrap
        """

        quote_page1 = input_url + '&pageIndex='
        quote_page2 = '&sortby=3&conditionFilter=-500'
    
        num_pages = pages
        review_list = []

        for i in range(num_pages):
            url = quote_page1 + str(i) + quote_page2
            page = requests.get(url)
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
                    review_list.append({'comment': comment, 'rating': calculated_rating})

        with open(self.output_path, 'w') as output_file:
            dict_writer = csv.DictWriter(output_file, ['comment', 'rating'])
            dict_writer.writeheader()
            dict_writer.writerows(review_list)

        print('Reviews scraped: ' + str(len(review_list)))
