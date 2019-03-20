"""Scrapes drugratingz.com for drug reviews.
"""

import csv
import requests
from bs4 import BeautifulSoup


class DrugRatingzScraper():
    """Scrapes drugratingz.com for drug reviews.
    """

    def scrape(self, drug_url):
        """Scrape for drug reviews.

        Args:
            drug_url: Drugsratingz.com page to scrape
            output_path: Path to the file where the output should be sent
        """

        page = requests.get(drug_url)
        soup = BeautifulSoup(page.text, 'html.parser')
        comments = [comment.text.strip() for comment in soup.find_all(
            'span', {'class': 'description'})]
        ratings = [rating.text.strip() for rating in soup.find_all(
            'td', {'align': 'center'}) if 'valign' in rating.attrs
            and rating.text.strip().isdigit()]

        review_list = []
        ratings_index = 0
        comment_index = 0

        while ratings_index < len(ratings):
            effectiveness = ratings[ratings_index]
            nosideeffects = ratings[ratings_index + 1]
            convenience = ratings[ratings_index + 2]
            value = ratings[ratings_index + 3]

            review_list.append({
                'comment': comments[comment_index],
                'effectiveness': effectiveness,
                'no side effects': nosideeffects,
                'convenience': convenience,
                'value': value
            })

            ratings_index = ratings_index + 4
            comment_index = comment_index + 1

        print('Reviews scraped: ' + str(len(review_list)))
        return review_list
