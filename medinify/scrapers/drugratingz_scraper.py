"""
Scrapes drugratingz.com for drug reviews.
"""

import re
import requests
from bs4 import BeautifulSoup
from medinify.scrapers.scraper import Scraper
import pandas as pd


class DrugRatingzScraper(Scraper):
    """Scrapes drugratingz.com for drug reviews.
    """

    def scrape_page(self, url):
        """
        Scrapes a single page of drug reviews
        :param url: drug reviews page url
        :return:
        """
        page = requests.get(url)
        soup = BeautifulSoup(page.text, 'html.parser')
        drug_name = soup.find('title').text.split()[0]
        reviews = [x for x in soup.find_all('tr', {'class': 'ratingstableodd'})
                   if 'class' not in x.find('td').attrs] + \
                  [x for x in soup.find_all('tr', {'class', 'ratingstableeven'})
                   if 'class' not in x.find('td').attrs]

        rows = {'comment': []}
        if 'rating' in self.data_collected:
            rows['rating'] = []
        if 'date' in self.data_collected:
            rows['date'] = []
        if 'drug' in self.data_collected:
            rows['drug'] = []
        if 'user id' in self.data_collected:
            raise AttributeError('DrugRatingz.com does to contain user ids.')
        if 'url' in self.data_collected:
            rows['url'] = []

        for review in reviews:
            rows['comment'].append(re.sub('\n+|\r+', '', review.find(
                'span', {'class': 'description'}).text.strip()))
            if 'rating' in self.data_collected:
                rating_types = ['effectiveness', 'no side effects', 'convenience', 'value']
                nums = [int(x.text.strip()) for x in review.find_all('td', {'align': 'center'}) if not x.find('img')]
                ratings = dict(zip(rating_types, nums))
                rows['rating'].append(ratings)
            if 'date' in self.data_collected:
                date = [x.text.strip().replace(u'\xa0', u' ') for x in review.find_all(
                    'td', {'valign': 'top'}) if not x.find('a') and 'align' not in x.attrs][0]
                rows['date'].append(date)
            if 'drug' in self.data_collected:
                rows['drug'].append(drug_name)
            if 'url' in self.data_collected:
                rows['url'].append(url)

        scraped_data = pd.DataFrame(rows, columns=self.data_collected)
        self.dataset = self.dataset.append(scraped_data, ignore_index=True)

    def scrape(self, url):
        """
        Scrapes all reviews of a given drug
        :param url: drug reviews url
        """
        self.scrape_page(url)

    def get_url(self, drug_name):
        """
        Given a drug name, finds the drug review page(s) on a given review forum
        :param drug_name: name of drug being searched for
        :return: drug url on given review forum
        """
        search_url = 'https://www.drugratingz.com/searchResults.jsp?thingname=' + \
                     drug_name.lower().split()[0] + '&1=&2='
        search_page = requests.get(search_url)
        search_soup = BeautifulSoup(search_page.text, 'html.parser')
        version_urls = []
        if search_soup.find('td', {'align': 'center'}):
            tags = search_soup.find_all('td', {'valign': 'middle'})

            version_urls = []
            for tag in tags:
                if tag.find('a') and 'reviews' in tag.find('a').attrs['href']:
                    tag_urls = tag.find_all('a')
                    for element in tag_urls:
                        if 'reviews' in element.attrs['href']:
                            version_urls.append('https://www.drugratingz.com' + element.attrs['href'])

            version_urls = list(set(version_urls))
        return version_urls






