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
from medinify.scrapers import WebMDScraper, EverydayHealthScraper, \
    DrugRatingzScraper, DrugsScraper

class ReviewDataset():
    """Dataset for collection, storing, and cleansing of drug reviews.

    Attributes:
        reviews: List of dictionaries with all review data
        drug_name: Name of drug this dataset was created for
    """
    reviews = []
    drug_name = ''
    meta = {'locked': False}
    scraper = None  # WebMD, EverydayHealth, Drugs, DrugRatingz

    def __init__(self, drug_name, scraper):
        drug_name = ''.join(drug_name.lower().split())
        drug_name = ''.join(char for char in drug_name if char.isalnum())
        self.drug_name = drug_name
        self.scraper = scraper
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

        scraper = None
        if self.scraper == 'WebMD':
            scraper = WebMDScraper()
        elif self.scraper == 'EverydayHealth':
            scraper = EverydayHealthScraper()
        elif self.scraper == 'Drugs':
            scraper = DrugsScraper()
        elif self.scraper == 'DrugRatingz':
            scraper = DrugRatingzScraper()

        if testing:
            scraper = WebMDScraper(False, 1)

        self.reviews = scraper.scrape(url)
        self.meta['endTimestamp'] = time()

    def collect_drug_names(self, file_path, output_path):
        """Given list of drug names, collect urls for those review page on
            the scraper's website

        Args:
            file_path: input csv with list of drug names
            output_path: output csv with urls
        """
        if self.scraper == 'WebMD':
            scraper = WebMDScraper()
            scraper.get_drug_urls(file_path, output_path)
        elif self.scraper == 'EverydayHealth':
            scraper = EverydayHealthScraper()
            scraper.get_drug_urls(file_path, output_path)
        elif self.scraper == 'Drugs':
            scraper = DrugsScraper()
            scraper.get_drug_urls(file_path, output_path)
        elif self.scraper == 'DrugRatingz':
            scraper = DrugRatingzScraper()
            scraper.get_drug_urls(file_path, output_path)

    def collect_urls(self, file_path, start=0):
        """Scrape all reviews for all drugs urls in file

        Args:
            start: index to start at if continuing from previous run
        """
        if self.meta['locked']:
            print('Dataset locked. Please load a different dataset.')
            return

        scraper = None
        if self.scraper == 'WebMD':
            scraper = WebMDScraper()
        elif self.scraper == 'EverydayHealth':
            scraper = EverydayHealthScraper()
        elif self.scraper == 'Drugs':
            scraper = DrugsScraper()
        elif self.scraper == 'DrugRatingz':
            scraper = DrugRatingzScraper()

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

    @staticmethod
    def collect_all_nanodrugs(drugname_input_file):
        """Collect all reviews for all nano drugs across WebMD, Drugs.com, and DrugRatingsz

        Args:
            drugname_input_file: File with list of drug names
        """
        webmd_dataset = ReviewDataset('webmd_nano', 'WebMD')
        drugs_dataset = ReviewDataset('drugs_nano', 'Drugs')
        drugratingz_dataset = ReviewDataset('drugratingz_nano', 'DrugRatingz')
        everydayhealth_dataset = ReviewDataset('everyday_nano', 'EverydayHealth')

        webmd_dataset.collect_drug_names(drugname_input_file, 'nano_webmd.csv')
        drugs_dataset.collect_drug_names(drugname_input_file, 'nano_drugs.csv')
        drugratingz_dataset.collect_drug_names(drugname_input_file, 'nano_drugratingz.csv')
        everydayhealth_dataset.collect_drug_names(drugname_input_file, 'nano_everydayhealth.csv')

        webmd_dataset.collect_urls('nano_webmd.csv')
        webmd_dataset.remove_empty_comments()
        webmd_dataset.generate_ratings()
        drugs_dataset.collect_urls('nano_drugs.csv')
        drugs_dataset.remove_empty_comments()
        drugs_dataset.generate_ratings_drugs()
        drugratingz_dataset.collect_urls('nano_drugratingz.csv')
        drugratingz_dataset.remove_empty_comments()
        drugratingz_dataset.generate_ratings_drugratingz()
        everydayhealth_dataset.collect_urls('nano_everydayhealth.csv')
        everydayhealth_dataset.remove_empty_comments()

        webmd_dataset.write_file('csv', 'nano_reviews_webmd.csv')
        drugs_dataset.write_file('csv', 'nano_reviews_drugs.csv')
        drugratingz_dataset.write_file('csv', 'nano_reviews_drugratingz.csv')
        everydayhealth_dataset.write_file('csv', 'nano_reviews_everydayhealth.csv')

        os.remove('nano_webmd.csv')
        os.remove('nano_drugs.csv')
        os.remove('nano_drugratingz.csv')
        os.remove('nano_everydayhealth.csv')

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

    def generate_ratings(self):
        """Generate final rating based on config file
        """
        updated_reviews = []

        with open('../dataset-settings.json', 'r') as config_file:
            config = json.load(config_file)

        if self.scraper == 'WebMD':
            using_rating = config['reviews']['webmd']['use_rating']
            ratings_being_used = 0

            # Counter number of ratings being combined
            for type_of_rating in using_rating:
                if using_rating[type_of_rating]:
                    ratings_being_used += 1

            for review in self.reviews:
                rating = 0

                # If the rating is being used, add it to the final rating, else remove it
                for type_of_rating, using in using_rating.items():
                    if using:
                        rating += review[type_of_rating]
                    del review[type_of_rating]

                # Get average of ratings being used
                # TODO (Jorge): Change all logic to use floats
                review['rating'] = int(rating / ratings_being_used)

                updated_reviews.append(review)

            self.reviews = updated_reviews

        # Rest of scrapers have not been updated to use config yet
        elif self.scraper == 'DrugRatingsz':
            self.generate_ratings_drugratingz()

        elif self.scraper == 'Drugs':
            self.generate_ratings_drugs()

        else:
            raise ValueError('Scraper "{}" does not exist'.format(self.scraper))

    def generate_ratings_drugratingz(self):
        """Generate rating based for drugratingz
        """
        updated_reviews = []

        if self.scraper == 'DrugRatingz':
            for review in self.reviews:
                review['rating'] = review['effectiveness']
                del review['effectiveness']
                updated_reviews.append(review)

            self.reviews = updated_reviews

    def generate_ratings_drugs(self):
        """Generate rating based for drugs
        """
        updated_reviews = []

        if self.scraper == 'Drugs':
            for review in self.reviews:
                review['rating'] = review['rating'] / 2.0
                updated_reviews.append(review)

            self.reviews = updated_reviews

    def print_stats(self):
        """Print relevant stats about the dataset
        """
        if self.scraper in ['WebMD', 'EverydayHealth']:
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
        else:
            print('print_stats not implemented for {} scraper'.format(self.scraper))

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
