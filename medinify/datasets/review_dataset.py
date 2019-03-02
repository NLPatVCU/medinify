"""
Dataset for collection, storing, and cleansing of drug reviews.
"""

import os
from time import time
from datetime import date
from datetime import datetime
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
    meta = {'locked': False}

    def __init__(self, drug_name):
        drug_name = ''.join(drug_name.lower().split())
        drug_name = ''.join(char for char in drug_name if char.isalnum())
        self.drug_name = drug_name
        print('Created object for {}'.format(self.drug_name))

    def collect(self, url, testing=False):
        """Scrapes drug reviews and saves them as dictionary property

        Args:
            url: WebMD URL where all the reviews are
        """
        if self.meta['locked']:
            print('Dataset locked. Please load a different dataset.')
            return

        self.meta['startTimestamp'] = time()
        self.meta['drugs'] = [self.drug_name]
        scraper = WebMDScraper()

        if testing:
            scraper = WebMDScraper(False, 1)

        self.reviews = scraper.scrape(url)
        self.meta['endTimestamp'] = time()

    def collect_urls(self, file_path, start=0):
        """Scrape all reviews for all drugs urls in file

        Args:
            start: index to start at if continuing from previous run
        """
        if self.meta['locked']:
            print('Dataset locked. Please load a different dataset.')
            return
        scraper = WebMDScraper()
        urls = []

        with open(file_path) as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                if row['URL'] != 'Not found':
                    urls.append({'name': row['Drug'], 'url': row['URL']})
        print('Found {} urls.'.format(len(urls)))

        if os.path.isfile(self.drug_name.lower() + '-dataset.pickle'):
            self.load()
        else:
            print('Saving meta...')
            drug_names = [x['name'] for x in urls]
            self.meta['drugs'] = drug_names
            self.meta['startTimestamp'] = time()
            self.save()

        # Loop through urls starting at start index
        for i in range(start, len(urls)):
            drug = urls[i]
            print('\n{} drugs left to scrape.'.format(len(urls) - i))
            print('Scraping {}...'.format(drug['name']))
            reviews = scraper.scrape(drug['url'])

            # If it's the first drug then replace self.reviews instead of appending
            if drug['name'] == urls[0]['name']:
                self.reviews = reviews
            else:
                self.reviews += reviews

            # Save our progress and let the user know the data is safe
            self.meta['endTimestamp'] = time()
            self.save()
            print('{} reviews saved. Safe to quit.'.format(drug['name']))

            # Let the user know what start index to use to continue later
            if i < len(urls) - 1:
                print('To continue run with parameter start={}'.format(i + 1))

        print('\nAll urls scraped!')

    def save(self):
        """Saves current reviews as a pickle file
        """
        if self.meta['locked']:
            print('Dataset locked. Please load a different dataset.')
            return

        filename = self.drug_name + '-dataset.pickle'
        data = {'meta': self.meta, 'reviews': self.reviews}
        with open(filename, 'wb') as pickle_file:
            pickle.dump(data, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

    def final_save(self):
        """Save current reviews as a pickle file with timestamp and locks it
        """
        if self.meta['locked']:
            print('Dataset locked. Please load a different dataset.')
            return
        self.meta['locked'] = True
        data = {'meta': self.meta, 'reviews': self.reviews}
        today = date.today()
        filename = self.drug_name + '-dataset-' + str(today) + '.pickle'

        with open(filename, 'wb') as pickle_file:
            pickle.dump(data, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, filename=None):
        """Loads set of reviews from a pickle file
        """
        if filename is None:
            filename = self.drug_name + '-dataset.pickle'

        with open(filename, 'rb') as pickle_file:
            data = pickle.load(pickle_file)
            self.reviews = data['reviews']
            self.meta = data['meta']

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

        print('Writing {}...'.format(filename))

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

        print('{} empty comments removed.'.format(empty_comments_removed))
        self.reviews = updated_reviews

    def generate_rating(self):
        """Generate rating based on source and options
        """
        updated_reviews = []

        for review in self.reviews:
            review['rating'] = review['effectiveness']
            del review['effectiveness']
            updated_reviews.append(review)

        self.reviews = updated_reviews

    def print_stats(self):
        """Print relevant stats about the dataset
        """
        reviews_ratings = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

        for review in self.reviews:
            rating = review['rating']
            reviews_ratings[rating] += 1

        print('\nTotal reviews: {}'.format(len(self.reviews)))
        for key, val in reviews_ratings.items():
            print('{} star ratings: {}'.format(key, val))

        positive_ratings = reviews_ratings[4] + reviews_ratings[5]
        negative_ratings = reviews_ratings[1] + reviews_ratings[2]
        print('Positive ratings: {}'.format(positive_ratings))
        print('Negative ratings: {}'.format(negative_ratings))
        print('Pos:Neg ratio: {}'.format(positive_ratings / negative_ratings))

    def print_reviews(self):
        """Prints out current dataset in human readable format
        """
        print('\n-----"{}" Review Dataset-----'.format(self.drug_name))
        pprint.pprint(self.reviews)
        print('\n"{}" Reviews: {}'.format(self.drug_name, len(self.reviews)))

    def print_meta(self):
        """Prints out meta data about dataset
        """
        locked = str(self.meta['locked'])
        start_timestamp = self.meta['startTimestamp']
        start_timestamp = datetime.utcfromtimestamp(start_timestamp).strftime('%Y-%m-%d %H:%M:%S')
        end_timestamp = self.meta['endTimestamp']
        end_timestamp = datetime.utcfromtimestamp(end_timestamp).strftime('%Y-%m-%d %H:%M:%S')

        print('Locked: ' + locked)
        print('Started scrape at ' + start_timestamp + ' UTC')
        print('Finished scrape at ' + end_timestamp + ' UTC')
        print('Drugs in dataset: ' + str(self.meta['drugs']))
