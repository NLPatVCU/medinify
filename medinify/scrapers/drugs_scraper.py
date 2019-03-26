"""Scrapes Drugs.com for drug reviews.
"""

import csv
import requests
from bs4 import BeautifulSoup
import os

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
        table_footer = soup.find('table', {'class': 'data-list ddc-table-sortable'}).find('tfoot').find('tr').find_all('th')
        total_reviews = int(table_footer[2].get_text().split()[0])

        max_pages = total_reviews // 25
        if (total_reviews % 25 != 0):
            max_pages += 1

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
                comment = review.find('span').text.lstrip('"').rstrip('"').replace('\n', '')

                if review.find('div', {'class': 'rating-score'}):
                    rating = float(review.find('div', {'class': 'rating-score'}).text)
                
                review_list.append({'comment': comment, 'rating': rating})
 
    
        print('Reviews scraped: ' + str(len(review_list)))
        return review_list

    def get_drug_urls(self, file_path, output_file):
        """Given a list of drug names, gets reviews pages on Drugs.com"""

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

