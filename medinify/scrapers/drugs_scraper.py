"""Scrapes Drugs.com for drug reviews.
"""

import csv
import requests
from bs4 import BeautifulSoup
import os

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
                review_url = 'https://www.drugs.com/comments/' + drug.lower().split()[0]
                review_page = requests.get(review_url)
                if review_page.status_code < 400:
                    drug_review_urls[drug] = review_url
                else:
                    search_url = 'https://www.drugs.com/' + drug.lower().split()[0] + '.html'
                    info_page = requests.get(search_url)
                    if info_page.status_code < 400:
                        info_soup = BeautifulSoup(info_page.text, 'html.parser')
                        if info_soup.find('span', {'class': 'ratings-total'}):
                            drug_review_urls[drug] = 'https://www.drugs.com' + info_soup.find(
                                'span', {'class': 'ratings-total'}).find('a').attrs['href']

            reverse_review_urls = dict((y, x) for x, y in drug_review_urls.items())
            no_duplicates_urls = dict((y, x) for x, y in reverse_review_urls.items())

            for drug in drugs:
                if drug not in list(no_duplicates_urls.keys()):
                    unfound_drugs.append(drug)

                drugs = list(no_duplicates_urls.keys())
                for drug in drugs:
                    entry = {'Drug': drug, 'URL': no_duplicates_urls[drug]}
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
