"""Scrapes drugratingz.com for drug reviews.
"""

import csv
import requests
from bs4 import BeautifulSoup
import os


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

    def get_drug_urls(self, file_path, output_file):

        drugs = []
        with open(file_path, 'r') as drug_names:
            drugs_reader = csv.reader(drug_names)
            for row in drugs_reader:
                drugs.append(row[0])

        # search for drug info pages
        unfound_drugs = []
        drug_review_urls = {}
        review_urls = []

        if not os.path.exists('drug_results.pickle'):
            for drug in drugs:
                print('Searching for {}'.format(drug))
                search_url = 'https://www.drugratingz.com/searchResults.jsp?thingname=' + \
                             drug.lower().split()[0] + '&1=&2='
                search_page = requests.get(search_url)
                search_soup = BeautifulSoup(search_page.text, 'html.parser')
                if search_soup.find('td', {'align': 'center'}):
                    tags = search_soup.find_all('td', {'valign': 'middle'})

                    if not tags:
                        unfound_drugs.append(drug)
                        continue

                    version_urls = []
                    for tag in tags:
                        if tag.find('a') and 'reviews' in tag.find('a').attrs['href']:
                            tag_urls = tag.find_all('a')
                            for element in tag_urls:
                                if 'reviews' in element.attrs['href']:
                                    version_urls.append('https://www.drugratingz.com' + element.attrs['href'])

                    version_urls = list(set(version_urls))
                    num_versions = len(version_urls)
                    if num_versions == 1:
                        drug_review_urls[drug] = version_urls[0]
                    elif num_versions > 1:
                        for i in range(num_versions):
                            drug_review_urls[drug + ' ' + str(i + 1)] = version_urls[i]

            drugs = list(drug_review_urls.keys())
            for drug in drugs:
                entry = {'Drug': drug, 'URL': drug_review_urls[drug]}
                review_urls.append(entry)

        print(str(len(unfound_drugs)) + ' drugs not found')
        print(unfound_drugs)

        # writes url csv file
        with open(output_file, 'w') as url_csv:
            fieldnames = ['Drug', 'URL']
            writer = csv.DictWriter(url_csv, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(review_urls)

        print('Finished writing!')






