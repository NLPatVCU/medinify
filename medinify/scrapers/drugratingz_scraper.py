"""Scrapes drugratingz.com for drug reviews.
"""

import csv
import requests
from bs4 import BeautifulSoup

class DrugRatingzScraper():
    """Scrapes drugratingz.com for drug reviews.
    """

    def scrape(self, drug_url, output_path):
        """Scrape for drug reviews.

        Args:
            drug_url: Drugsratingz.com page to scrape
            output_path: Path to the file where the output should be sent
        """

        page = requests.get(drug_url)
        soup = BeautifulSoup(page.text, 'html.parser')
        comments = soup.find_all('span', {'class' : 'description'})
        ratings = soup.find_all('td', {'align' : 'center'})
        ratings = [rating for rating in ratings if "height" not in rating.attrs and 'rowspan' not in rating.attrs
                   and not rating.find('a') and not rating.find('img')]

        review_list = []
        ratings_index = 0
        comment_index = 0

        while ratings_index < len(ratings):
            effectiveness = ratings[ratings_index].text.strip()
            nosideeffects = ratings[ratings_index + 1].text.strip()
            convenience = ratings[ratings_index + 2].text.strip()
            value = ratings[ratings_index + 3].text.strip()

            review_list.append({'comment' : comments[comment_index].text.strip(), 'effectiveness' : effectiveness,
                                'no side effects' : nosideeffects, 'convenience' : convenience, 'value' : value})
            ratings_index = ratings_index + 4
            comment_index = comment_index + 1

        with open(output_path, 'w') as output_file:
            dict_writer = csv.DictWriter(output_file, ['comment', 'effectiveness', 'no side effects', 'convenience', 'value'])
            dict_writer.writeheader()
            dict_writer.writerows(review_list)

        print('Reviews scraped: ' + str(len(review_list)))
