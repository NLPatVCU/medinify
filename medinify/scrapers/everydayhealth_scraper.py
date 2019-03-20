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
                    rating = re.sub('Stars', '', rating)

                review_list.append({'comment': comment, 'for': review_for, 'rating': rating})

        print("Number of reviews scraped: " + str(len(review_list)))
        return review_list

