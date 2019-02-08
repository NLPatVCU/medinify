"""
Dataset for collection, storing, and cleansing of drug reviews.
"""

import pickle
import csv
import json
import pprint
from random import shuffle
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

    def collect(self, url, testing=False):
        """Scrapes drug reviews and saves them as dictionary property

        Args:
            url: WebMD URL where all the reviews are
        """
        # TODO(Jorge): Remove need for url variable by pulling urls from stored file
        # TODO(Jorge): Add parameter for selecting which source
        scraper = WebMDScraper()

        if testing:
            scraper = WebMDScraper(False, 1)

        self.reviews = scraper.scrape(url)

    def collect_all_common_reviews(self, start=0):
        """Scrape all reviews for all "common" drugs on main WebMD drugs page

        Args:
            start: index to start at if continuing from previous run
        """
        # Load in case we have pre-exisiting progress
        self.load()
        scraper = WebMDScraper()

        # Get common drugs names and urls
        common_drugs = scraper.get_common_drugs()
        print(f'Found {len(common_drugs)} common drugs.')

        # Loop through common drugs starting at start index
        for i in range(start, len(common_drugs)):
            drug = common_drugs[i]
            print(f'\n{len(common_drugs) - i} drugs left to scrape.')
            print(f'Scraping {drug["name"]}...')
            reviews = scraper.scrape(drug['url'])

            # If it's the first drug then replace self.reviews instead of appending
            if drug['name'] == 'Actos':
                self.reviews = reviews
            else:
                self.reviews += reviews

            # Save our progress and let the user know the data is safe
            self.save()
            print(f'{drug["name"]} reviews saved. Safe to quit.')

            # Let the user know what start index to use to continue later
            if i < len(common_drugs) - 1:
                print(f'To continue run with parameter start={i + 1}')

        print('\nAll common drug review scraping complete!')

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

    # def combine_ratings(self, effectiveness=True, ease=True, satisfaction=True):
    #     """Take 3 WebMD ratings, save the average, and remove the original scores
    #     """
    #     updated_reviews = []
    #     types_of_ratings = sum([effectiveness, ease, satisfaction])

    #     print('Combining ratings...')

    #     for review in self.reviews:
    #         rating_sum = 0

    #         rating_sum += review['effectiveness'] if effectiveness else 0
    #         rating_sum += review['ease of use'] if ease else 0
    #         rating_sum += review['satisfaction'] if satisfaction else 0

    #         del review['effectiveness']
    #         del review['ease of use']
    #         del review['satisfaction']

    #         review['rating'] = float(rating_sum) / float(types_of_ratings)
    #         updated_reviews.append(review)

    #     self.reviews = updated_reviews

    def generate_rating(self):
        """Generate rating based on source and options
        """
        updated_reviews = []

        for review in self.reviews:
            review['rating'] = review['effectiveness']
            del review['effectiveness']
            updated_reviews.append(review)

        self.reviews = updated_reviews

    def balance(self):
        """Remove ratings so there's even number of positive and negative
        """
        positive_reviews = []
        negative_reviews = []

        for review in self.reviews:
            if review['rating'] == 5:
                positive_reviews.append(review)
            elif review['rating'] <= 2:
                negative_reviews.append(review)

        positives = len(positive_reviews)
        negatives = len(negative_reviews)
        
        least_reviews = min([positives, negatives])

        shuffle(positive_reviews)
        positive_reviews = positive_reviews[:least_reviews]
        shuffle(negative_reviews)
        negative_reviews = negative_reviews[:least_reviews]

        self.reviews = positive_reviews + negative_reviews

    def print_stats(self):
        """Print relevant stats about the dataset
        """
        reviews_ratings = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

        for review in self.reviews:
            rating = review['effectiveness']
            reviews_ratings[rating] += 1

        print(f'\nTotal reviews: {len(self.reviews)}')
        for key, val in reviews_ratings.items():
            print(f'{key} star ratings: {val}')

        positive_ratings = reviews_ratings[5]
        negative_ratings = reviews_ratings[1] + reviews_ratings[2]
        print(f'Positive ratings: {positive_ratings}')
        print(f'Negative ratings: {negative_ratings}')
        print(f'Pos:Neg ratio: {positive_ratings / negative_ratings}')

    def print_reviews(self):
        """Prints out current dataset in human readable format
        """
        print(f'\n-----"{self.drug_name}" Review Dataset-----')
        pprint.pprint(self.reviews)
        print(f'\n"{self.drug_name}" Reviews: {len(self.reviews)}')
