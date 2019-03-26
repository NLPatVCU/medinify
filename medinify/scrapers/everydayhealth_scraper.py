"""
EverydayHealth.com drug review scraper
"""
import csv
import re
import requests
from bs4 import BeautifulSoup
import os

class EverydayHealthScraper():
    """Scrapes EverydayHealth.com for drug reviews.
    """

    def scrape(self, url, pages=1):  # fix pages setup once added max pages method
        """Scrape for drug reviews.

        Args:
            url: EverydayHealth.com page to scrape
            pages (int): Number of pages to scrape
        """

        review_list = []

        for i in range(pages):
            new_url = url + '?page=' + str(i+1)
            page = requests.get(new_url)
            soup = BeautifulSoup(page.text, 'html.parser')
            reviews = soup.find_all('div', {'class': 'review-container row'})

            for review in reviews:
                review_for = review.find('h3').text.lstrip('"').rstrip('"')
                review_for = re.sub('Report', '', review_for)

                comment = review.find('p').text.lstrip('"').rstrip('"')
                comment = re.sub('Report', '', comment)

                if review.find('div', {'class': 'star-rating-print'}):
                    rating = review.find('div', {'class': 'star-rating-print'}).text
                    rating = float(re.sub('Stars', '', rating).strip())

                review_list.append({'comment': comment, 'for': review_for, 'rating': rating})

        print("Number of reviews scraped: " + str(len(review_list)))
        return review_list

    def get_drug_urls(self, file_path, output_file):
        """Given a list of drug names, gets reviews pages on EverydayHealth.com"""

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
                review_url = 'https://www.everydayhealth.com/drugs/' + drug.lower().split()[0] + '/reviews'
                review_page = requests.get(review_url)
                if review_page.status_code < 400:
                    drug_review_urls[drug] = review_url
                else:
                    search_url = 'https://www.everydayhealth.com/search/' + drug
                    search_page = requests.get(search_url)
                    search_soup = BeautifulSoup(search_page.text, 'html.parser')
                    if search_soup.find('div', {'class': 'resultTitle'}):
                        drug_info_url = 'https:' + search_soup.find('div', {'class': 'resultTitle'}).find(
                            'a').attrs['href']
                        drug_info_page = requests.get(drug_info_url)
                        drug_info_soup = BeautifulSoup(drug_info_page.text, 'html.parser')
                        if drug_info_soup.find('div', {'class': 'review-links drugs-profile__review-links'}):
                            drug_review_url = 'https://www.everydayhealth.com' + drug_info_soup.find(
                                'div', {'class': 'review-links drugs-profile__review-links'}).find('a').attrs['href']
                            drug_review_urls[drug] = drug_review_url

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
