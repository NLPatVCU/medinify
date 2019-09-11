"""
Scrapes drugratingz.com for drug reviews.
"""

import requests
from bs4 import BeautifulSoup
from medinify.scrapers.scraper import Scraper
import pandas as pd


class DrugRatingzScraper(Scraper):
    """Scrapes drugratingz.com for drug reviews.
    """

    def __init__(self, collect_user_ids=False, collect_urls=False):
        super(DrugRatingzScraper, self).__init__(collect_user_ids, collect_urls)
        if collect_user_ids:
            raise AttributeError('DrugRatingz.com does not collect user id data')

    def scrape_page(self, url):
        """
        Scrapes a single page of drug reviews
        :param url: drug reviews page url
        :return:
        """
        assert url[:36] == 'https://www.drugratingz.com/reviews/', 'Invalid url'

        page = requests.get(url)
        soup = BeautifulSoup(page.text, 'html.parser')
        drug_name = soup.find('title').text.split()[0]
        reviews = [x for x in soup.find_all('tr', {'class': 'ratingstableodd'})
                   if 'class' not in x.find('td').attrs] + \
                  [x for x in soup.find_all('tr', {'class', 'ratingstableeven'})
                   if 'class' not in x.find('td').attrs]

        rows = {'comment': [], 'rating': [], 'date': [], 'drug': []}
        if 'url' in self.data_collected:
            rows['url'] = []

        for review in reviews:
            rows['comment'].append(review.find('span', {'class': 'description'}).text.strip())

            rating_types = ['effectiveness', 'no side effects', 'convenience', 'value']
            nums = [int(x.text.strip()) for x in review.find_all('td', {'align': 'center'}) if not x.find('img')]
            ratings = dict(zip(rating_types, nums))
            rows['rating'].append(ratings)

            date = [x.text.strip().replace(u'\xa0', u' ') for x in review.find_all(
                'td', {'valign': 'top'}) if not x.find('a') and 'align' not in x.attrs][0]
            rows['date'].append(date)
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
        if not drug_name or len(drug_name) < 4:
            print('{} name too short; Please manually search for such reviews'.format(drug_name))
            return []

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






