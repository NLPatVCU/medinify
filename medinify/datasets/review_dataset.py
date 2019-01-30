"""
Dataset for collection, storing, and cleansing of drug reviews.
"""

import pickle
import csv
import json
import pprint
from medinify.scrapers import WebMDScraper


class ReviewDataset():
    """Dataset for collection, storing, and cleansing of drug reviews.

    Attributes:
        reviews: List of dictionaries with all review data
        drug_name: Name of drug this dataset was created for
    """
    reviews = []
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
        pprint.pprint(self.reviews)
        print(f'\n"{self.drug_name}" Reviews: {len(self.reviews)}')

    def collect(self, url, testing=False):
        """Scrapes drug reviews and saves them as dictionary property

        Args:
            url: WebMD URL where all the reviews are
        """
        # TODO(Jorge): Remove need for url variable by pulling urls from stored file
        # TODO(Jorge): Add parameter for selecting which source or all
        scraper = WebMDScraper()

        if testing:
            scraper = WebMDScraper(False, 2)

        self.reviews = scraper.scrape(url)

    def save(self):
        """Saves current reviews as a pickle file
        """
        filename = self.drug_name + '-dataset.pickle'
        with open(filename, 'wb') as pickle_file:
            pickle.dump(
                self.reviews, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

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
            filetype: Type of file to save data as (csv, json)
            filename: Name of file to save CSV as
        """

        filetype = filetype.lower()

        if filetype not in ('csv', 'json'):
            raise ValueError('Filetype needs to be "csv" or "json"')

        if filename is None:
            filename = self.drug_name + '-reviews.' + filetype

        print(f'Writing {filename}...')

        if filetype == 'csv':
            with open(filename, 'w') as output_file:
                dict_writer = csv.DictWriter(output_file,
                                             self.reviews[0].keys())
                dict_writer.writeheader()
                dict_writer.writerows(self.reviews)
        elif filetype == 'json':
            with open(filename, 'w') as output_file:
                json.dump(self.reviews, output_file, indent=4)

        print('Done!')

    def remove_empty_comments(self):
        """Remove reviews with empty comments
        """
        updated_reviews = []
        empty_comments_removed = 0

        print('Removing empty comments...')

        for review in self.reviews:
            if review['comment']:
                updated_reviews.append(review)
            else:
                empty_comments_removed += 1

        print(f'{empty_comments_removed} empty comments removed.')
        self.reviews = updated_reviews

    def combine_ratings(self):
        """Take 3 WebMD ratings, save the average, and remove the original scores
        """
        updated_reviews = []

        print('Combining ratings...')

        for review in self.reviews:
            rating_sum = 0

            rating_sum += review['effectiveness']
            rating_sum += review['ease of use']
            rating_sum += review['satisfaction']

            del review['effectiveness']
            del review['ease of use']
            del review['satisfaction']

            review['rating'] = rating_sum / 3.0
            updated_reviews.append(review)

        self.reviews = updated_reviews

    def cleanse(self):
        """Run all cleansing functions
        """
        self.remove_empty_comments()
        self.combine_ratings()
        print('Done!')
