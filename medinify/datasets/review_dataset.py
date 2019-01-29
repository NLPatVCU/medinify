"""
Dataset for collection, storing, and cleansing of drug reviews.
"""

import pickle
import csv
import json
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

    def write_file(self, filetype, filename=None):
        """Creates CSV file of review data

        Args:
            filetype: Type of file to save data as
            filename: Name of file to save CSV as
        """
        # TODO: Error checking for filetype that isn't csv or json

        if filename is None:
            filename = self.drug_name + '-reviews.' + filetype

        print(f'Writing {filename}...')

        if filetype == 'csv':
            with open(filename, 'w') as output_file:
                dict_writer = csv.DictWriter(output_file, ['comment', 'rating'])
                # TODO: Set the header based on dictionary keys
                dict_writer.writeheader()
                dict_writer.writerows(self.reviews)
                print('Done!')
        elif filetype == 'json':
            with open(filename, 'w') as output_file:
                json.dump(self.reviews, output_file)
