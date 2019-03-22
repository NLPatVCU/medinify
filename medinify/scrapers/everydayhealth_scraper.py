"""
EverydayHealth.com drug review scraper
"""
import csv
import re
import requests
from bs4 import BeautifulSoup

class EverydayHealthScraper():
    """Scrapes EverydayHealth.com for drug reviews.
    """

    def scrape(self, url, output_path, pages):
        """Scrape for drug reviews.

        Args:
            drug_url: EverydayHealth.com page to scrape
            output_path: Path to the file where the output should be sent
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
                    rating = re.sub('Stars', '', rating)

                review_list.append({'comment': comment, 'for': review_for, 'rating': rating})

        with open(output_path, 'w') as output_file:
            dict_writer = csv.DictWriter(output_file, ['comment', 'for', 'rating'])
            dict_writer.writeheader()
            dict_writer.writerows(review_list)

        print("Number of reviews scraped: " + str(len(review_list)))

    def max_pages(self, input_url):
        """Finds number of review pages for this drug.
        Args:
            input_url: URL for the first page of reviews.
        Returns:
            (int) Highest page number
        """
        while True:

                page = requests.get(input_url)
                soup = BeautifulSoup(page.text, 'html.parser')

                # Case if no reviews available
                
                break

        total_reviews_head = soup.find('div', {'class': 'review-details clearfix'}).find('h5').find('span', {'itemprop': 'reviewCount'}).text
        total_reviews = int(total_reviews_head)

        max_pages_foot = soup.find('div', {'class': 'review-pagination'}).find('section', {'class': 'review-pagination__section--info'}).text.split()
        max_pages = int(max_pages_foot[2])
    
       
        print('Found ' + str(total_reviews) + ' reviews.')
        print('Scraping ' + str(max_pages) + ' pages...')
        return max_pages
