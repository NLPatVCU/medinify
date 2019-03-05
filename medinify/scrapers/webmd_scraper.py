"""
Drug review scraper for Medinify.
This module scrapes comments from WebMD along with their rating.
Based on work by Amy Olex 11/13/17.
"""

import re
from time import sleep
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup
import csv
import pickle
import os

class WebMDScraper():
    """
    Class to scrap drug reviews from WebMD
    Attributes:
        all_pages: Boolean for whether or not to scrape all pages
        pages: int for # of pages to scrape if all_pages is 0
        review_list: List of review dictionary items
    """

    all_pages = True
    pages = 1
    review_list = []

    def __init__(self, all_pages=True, pages=1):
        self.all_pages = all_pages
        self.pages = pages

    def max_pages(self, input_url):
        """Finds number of review pages for this drug.
        Args:
            input_url: URL for the first page of reviews.
        Returns:
            (int) Highest page number
        """


        while True:
            try:
                page = requests.get(input_url)
                soup = BeautifulSoup(page.text, 'html.parser')
                if 'Be the first to share your experience with this treatment.' in soup.find('div', {'id': 'heading'}).text:
                    return 0
                break
            except AttributeError:
                print('Ran into AttributeError. Waiting 10 seconds and retrying...')
                sleep(10)

        total_reviews_text = soup.find('span', {'class': 'totalreviews'}).text
        total_reviews = [int(s) for s in total_reviews_text.split() if s.isdigit()][0]

        # Does the equivalent of max_pages = ceil(total_reviews / 5) without the math library
        max_pages = total_reviews // 5
        if total_reviews % 5 != 0:
            max_pages += 1

        print('Found ' + str(total_reviews) + ' reviews.')
        print('Scraping ' + str(max_pages) + ' pages...')
        return max_pages

    def scrape_page(self, page_url):
        """Scrapes a single page for reviews and adds them to review_list
        Args:
            page_url: URL of the page to scrape.
        """
        page = requests.get(page_url)
        soup = BeautifulSoup(page.text, 'html.parser')
        reviews = soup.find_all('div', attrs={'class': 'userPost'})

        for review in reviews:
            comment = review.find('p', id=re.compile("^comFull*")).text
            comment = comment.replace('Comment:', '').replace('Hide Full Comment', '')
            comment = ' '.join(comment.splitlines())

            ratings = review.find_all('span', attrs={'class': 'current-rating'})
            effectiveness = int(ratings[0].text.replace('Current Rating:', '').strip())
            ease = int(ratings[1].text.replace('Current Rating:', '').strip())
            satisfaction = int(ratings[2].text.replace('Current Rating:', '').strip())

            self.review_list.append({'comment': comment,
                                     'effectiveness': effectiveness,
                                     'ease of use': ease,
                                     'satisfaction': satisfaction})

    def scrape(self, input_url):
        """Scrapes the reviews from WebMD
        Args:
            input_url : WebMD URL to scrape
        """

        print('Scraping WebMD...')

        self.review_list = []

        quote_page1 = input_url + '&pageIndex='
        quote_page2 = '&sortby=3&conditionFilter=-1'
        num_pages = 0

        if self.all_pages:
            num_pages = self.max_pages(input_url)
        else:
            num_pages = self.pages

        for i in range(num_pages):
            page_url = quote_page1 + str(i) + quote_page2
            self.scrape_page(page_url)

            page = i + 1
            if page % 10 == 0:
                print('Scraped ' + str(page) + ' pages...')

        print('Reviews scraped: ' + str(len(self.review_list)))

        return self.review_list

    def get_common_drugs(self):
        """ Get all urls for 'common' drug review pages
        Returns:
            List of urls for each drug's review page
        """
        url = 'https://www.webmd.com/drugs/2/index?show=drugs'
        page = requests.get(url)
        soup = BeautifulSoup(page.text, 'html.parser')
        drug_names = soup.find_all('a', {'class': 'common-result-name'})
        drug_review_links = soup.find_all('a', {'class': 'common-result-review'})
        drug_review_pages = []

        for i in range(1, len(drug_names)):
            name = drug_names[i].text
            relative_link = drug_review_links[i]['href']
            absolute_link = urljoin(url, relative_link)
            drug_review_pages.append({'name': name, 'url': absolute_link})

        return drug_review_pages

    def get_drug_urls(self, file_path, output_file):

        drugs = []
        with open(file_path, 'r') as drug_names:
            drugs_reader = csv.reader(drug_names)
            for row in drugs_reader:
                drugs.append(row[0])

        # search for drug info pages
        unfound_drugs = []
        drug_info_urls = {}

        if not os.path.exists('drug_results.pickle'):
            for drug in drugs:
                print('Searching for {}'.format(drug))
                search_url = 'https://www.webmd.com/drugs/2/search?type=drugs&query=' + drug.lower()
                search_page = requests.get(search_url)
                search_soup = BeautifulSoup(search_page.text, 'html.parser')
                if len(drug) < 4 and search_soup.find('ul', {'class': 'exact-match'}):
                    exact_matches = search_soup.find('ul', {'class': 'exact-match'})
                    search_results = exact_matches.find_all('a', {'data-metrics-link': 'result_1'})
                    links = ['https://www.webmd.com' + link.attrs['href'] for link in search_results]
                    print('\nMultiple versions were found for {}'.format(drug))
                    print('Its name was determined to be too short to verify the result\'s legitimacy')
                    print('The following versions were found:')
                    for link in links:
                        version_page = requests.get(link)
                        version_soup = BeautifulSoup(version_page.text, 'html.parser')
                        drug_name = version_soup.find('h1').text
                        print(drug_name)
                    print('\n')

                elif search_soup.find('a', {'class': 'drug-review'}):
                    drug_info_urls[drug] = search_url
                elif search_soup.find('ul', {'class': 'exact-match'}):
                    exact_matches = search_soup.find('ul', {'class': 'exact-match'})
                    search_results = exact_matches.find_all('a', {'data-metrics-link': 'result_1'})
                    links = ['https://www.webmd.com' + link.attrs['href'] for link in search_results]
                    for link in links:
                        version_page = requests.get(link)
                        version_soup = BeautifulSoup(version_page.text, 'html.parser')
                        drug_name = version_soup.find('h1').text
                        drug_info_urls[drug_name] = link
                else:
                    unfound_drugs.append(drug)

        print('Drugs not found: {}'.format(unfound_drugs))
        drugs = list(drug_info_urls.keys())

        # searches on drug info pages for drug review pages
        drug_review_pages = {}
        for drug in drugs:
            print('Searching for {} reviews page'.format(drug))
            drug_info = drug_info_urls[drug]
            drug_info_page = requests.get(drug_info)
            drug_info_soup = BeautifulSoup(drug_info_page.text, 'html.parser')
            if drug_info_soup.find('a', {'class': 'drug-review'}):
                print('Found {} reviews page\n'.format(drug))
                drug_review_page = 'https://www.webmd.com' + drug_info_soup.find('a', {'class': 'drug-review'}).attrs['href']
                if drug_review_page not in drug_review_pages.values():
                    drug_review_pages[drug] = drug_review_page

        print('Found urls for {} drugs'.format(len(drug_review_pages)))
        print('Did not find urls for {} drugs:\n {}\n'.format(len(unfound_drugs), unfound_drugs))

        print('Writing url csv file')
        review_urls = []
        for drug in drugs:
            review_urls.append({'Drug': drug, 'URL': drug_review_pages[drug]})

        # writes url csv file
        with open(output_file, 'w') as url_csv:
            fieldnames = ['Drug', 'URL']
            writer = csv.DictWriter(url_csv, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(review_urls)

        print('Finished writing!')
        if os.path.exists('drug_info_page.pickle'):
            os.remove('drug_info_page.pickle')
        if os.path.exists('drug_results.pickle'):
            os.remove('drug_results.pickle')
