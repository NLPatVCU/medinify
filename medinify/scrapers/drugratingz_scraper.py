"""
Drug review scraper for DrugRatingz.com
Implements the ability to collect the following data:

    -> Comments (Review text)
    -> 5-Point Scale Star Ratings ('Effectiveness', 'Convenience', 'No Side Effects', and 'Value')
    -> Post Dates
    -> Each review's associated url
    -> Each review's associated drug name

and to search for review urls given a drug name
"""
import requests
from bs4 import BeautifulSoup
from medinify.scrapers.scraper import Scraper
from tqdm import tqdm


class DrugRatingzScraper(Scraper):
    """
    The DrugRatingzScraper class implements drug review scraping functionality for DrugRatingz.com

    Attributes:
        collect_urls:    (Boolean) Whether or not to collect each review's associated url
        reviews:         (list[dict]) Scraped review data
    """

    nickname = 'drugratingz'

    def scrape_page(self, url):
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
            if self.collect_urls:
                row['url'] = url
            self.reviews.append(row)

    def scrape(self, url):
        """
        Scrapes all review for a given drug on DrugRatingz.com into 'reviews' attribute
        :param url: (str) url to the drug reviews for this drug
        """
        super().scrape(url)
        front_page = requests.get(url)
        front_page_soup = BeautifulSoup(front_page.text, 'html.parser')
        try:
            title = front_page_soup.find('h1').text
            assert 'drug reviews' in title
        except AssertionError:
            print('Invalid URL entered: %s' % url)
            return 0
        except AttributeError:
            print('Invalid URL entered: %s' % url)
            return 0
        self.scrape_page(url)

    def get_url(self, drug_name):
        """
        Searches DrugRatingz.com for reviews for a certain drug
        :param drug_name: (str) name of the drug to search for
        :return review_url: (str or None) if reviews for a drug with a matching name are found,
            this is the url for the first page of those reviews
            if a match was not found, returns None
        """
        if len(drug_name) < 4:
            print('%s name too short; Please manually search for such reviews' % drug_name)
            return None
        search_url = 'https://www.drugratingz.com/searchResults.jsp?thingname=' + \
                     drug_name.lower().split()[0] + '&1=&2='
        search_page = requests.get(search_url)
        search_soup = BeautifulSoup(search_page.text, 'html.parser')
        search_results = list(search_soup.find_all('tr', {'class': 'ratingstableeven'}) + search_soup.find_all('tr', {'class': 'ratingstableodd'}))

        max_reviews = -1
        max_index = -1

        for i, result in enumerate(search_results):
            num_reviews = int(list(result.find_all('td', {'align': 'center'}))[2].text.strip())
            if num_reviews > max_reviews:
                max_reviews = num_reviews
                max_index = i

        reviews_url = None
        if max_index > -1:
            reviews_url = 'https://www.drugratingz.com' + search_results[max_index].find_all(
                'td', {'align': 'center'})[2].find('a').attrs['href']

        return reviews_url







