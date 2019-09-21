"""
Scrapes drugratingz.com for drug reviews.
"""

import requests
from bs4 import BeautifulSoup
from medinify.scrapers.scraper import Scraper
from tqdm import tqdm


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
                   if x.find('span', {'class': 'description'})] + \
                  [x for x in soup.find_all('tr', {'class', 'ratingstableeven'})
                   if x.find('span', {'class': 'description'})]

        if len(reviews) == 0:
            print('No reviews found for drug %s' % drug_name)
            return

        print('Scraping DrugRatingz.com for %s Reviews...' % drug_name)
        for review in tqdm(reviews):
            row = {'comment': review.find('span', {'class': 'description'}).text.strip()}
            rating_types = ['effectiveness', 'no side effects', 'convenience', 'value']
            nums = [x for x in review.find_all('td', {'align': 'center'}) if 'valign'
                    in x.attrs and not x.find('a') and not x.find('img')]
            ratings = [int(x.text.replace(u'\xa0', u'')) for x in nums]
            row['rating'] = dict(zip(rating_types, ratings))

            row['date'] = [x.text.strip().replace(u'\xa0', u' ') for x in review.find_all(
                'td', {'valign': 'top'}) if not x.find('a') and 'align' not in x.attrs][0]
            row['drug'] = drug_name
            if 'url' in self.data_collected:
                row['url'] = url
            self.reviews.append(row)

    def scrape(self, url):
        """
        Scrapes all reviews of a given drug
        :param url: drug reviews url
        """
        super().scrape(url)
        self.scrape_page(url)

    def get_url(self, drug_name, return_multiple=False):
        """
        Given a drug name, finds the drug review page(s) on a given review forum
        :param drug_name: name of drug being searched for
        :param return_multiple: if multiple urls are found, whether or not to return all of them
        :return: drug url on given review forum
        """
        if not drug_name or len(drug_name) < 4:
            print('%s name too short; Please manually search for such reviews' % drug_name)
            return []

        search_url = 'https://www.drugratingz.com/searchResults.jsp?thingname=' + \
                     drug_name.lower().split()[0] + '&1=&2='
        search_page = requests.get(search_url)
        search_soup = BeautifulSoup(search_page.text, 'html.parser')
        review_urls = []
        if search_soup.find('td', {'align': 'center'}):
            tags = search_soup.find_all('td', {'valign': 'middle'})

            review_urls = []
            for tag in tags:
                if tag.find('a') and 'reviews' in tag.find('a').attrs['href']:
                    tag_urls = tag.find_all('a')
                    for element in tag_urls:
                        if 'reviews' in element.attrs['href']:
                            review_urls.append('https://www.drugratingz.com' + element.attrs['href'])

            review_urls = list(set(review_urls))
        if return_multiple and len(review_urls) > 0:
            print('Found %d Review Page(s) for %s' % (len(review_urls), drug_name))
            return review_urls
        elif len(review_urls) > 0:
            return review_urls[0]
        else:
            print('Found no %s reviews' % drug_name)
            return None






