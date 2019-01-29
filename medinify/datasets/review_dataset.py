"""
Dataset for collection, storing, and cleansing of drug reviews.
"""

import pickle
from medinify.scrapers import WebMDScraper

class ReviewDataset():
    """Dataset for collection, storing, and cleansing of drug reviews.
    """
    reviews = {}
    drug_name = ''

    def __init__(self, drug_name):
        drug_name = ''.join(drug_name.lower().split())
        drug_name = ''.join(char for char in drug_name if char.isalnum())
        self.drug_name = drug_name
        print(f'Created object for "{self.drug_name}"')

    def print(self):
        """Prints out current dataset in human readable format
        """
        print(f'\n-----"{self.drug_name}" Review Dataset-----')
        for review in self.reviews:
            print('Rating: ', review['rating'])
            print('Comment: ', review['comment'])
            print()
        print(f'"{self.drug_name}" Reviews: {len(self.reviews)}')

    def collect(self, url):
        """Scrapes drug reviews and saves them as dictionary property
        """
        # TODO: Remove need for url variable by pulling urls from stored file
        scraper = WebMDScraper(False, 1)
        self.reviews = scraper.scrape(url)


    def save(self):
        """Saves current reviews as a pickle file
        """
        filename = self.drug_name + '-dataset.pickle'
        with open(filename, 'wb') as pickle_file:
            pickle.dump(self.reviews, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self):
        """Loads set of reviews from a pickle file
        """
        filename = self.drug_name + '-dataset.pickle'
        with open(filename, 'rb') as pickle_file:
            data = pickle.load(pickle_file)
            self.reviews = data
