
import pandas as pd
import pickle
import os
import datetime
import numpy as np
import ast
from gensim.models import KeyedVectors
from medinify.scrapers.webmd_scraper import WebMDScraper
from medinify.scrapers.drugs_scraper import DrugsScraper
from medinify.scrapers.drugratingz_scraper import DrugRatingzScraper
from medinify.scrapers.everydayhealth_scraper import EverydayHealthScraper
from medinify.datasets.process.process import Processor
from medinify import config


class Dataset:
    """
    The Dataset class stores and processes data
    If also wraps around the functionality of the review scrapers for collecting review data

    Attributes:
        data_used: list of review data to collect (i.e., comment, rating, drug, date, etc.)
        data: DataFrame containing review data
        scraper: Which scraper to use for scraping functionality
        processor: The dataset's processor, defines how data is transformed into numeric representation
        start_timestamp: If collecting reviews, time when collection started
        end_timestamp: If collecting reviews, time when collection ended
        use_rating: whether or not to store rating data
        use_dates: whether or not to store date data
        use_drugs: whether or not to store drug name data
        use_user_ids: whether or not to store user id data
        use_urls: whether or not to store drug url data
    """
    scraper = None
    data_used = []

    def __init__(self, scraper=None,
                 use_rating=True, use_dates=True,
                 use_drugs=True, use_user_ids=False,
                 use_urls=False, w2v_file=None, pos=None,
                 pos_threshold=4.0, neg_threshold=2.0,
                 num_classes=2, rating_type='effectiveness',
                 data_representation='count'):
        if scraper == 'WebMD':
            self.scraper = WebMDScraper(collect_ratings=use_rating, collect_dates=use_dates,
                                        collect_drugs=use_drugs, collect_user_ids=use_user_ids,
                                        collect_urls=use_urls)
        elif scraper == 'Drugs':
            self.scraper = DrugsScraper(collect_ratings=use_rating, collect_dates=use_dates,
                                        collect_drugs=use_drugs, collect_user_ids=use_user_ids,
                                        collect_urls=use_urls)
        elif scraper == 'DrugRatingz':
            self.scraper = DrugRatingzScraper(collect_ratings=use_rating, collect_dates=use_dates,
                                              collect_drugs=use_drugs, collect_user_ids=use_user_ids,
                                              collect_urls=use_urls)
        elif scraper == 'EverydayHealth':
            self.scraper = EverydayHealthScraper(collect_ratings=use_rating, collect_dates=use_dates,
                                                 collect_drugs=use_drugs, collect_user_ids=use_user_ids,
                                                 collect_urls=use_urls)

        if self.scraper:
            self.data_used = self.scraper.data_collected
        else:
            self.data_used.append('comment')
            if use_rating:
                self.data_used.append('rating')
            if use_dates:
                self.data_used.append('date')
            if use_drugs:
                self.data_used.append('drug')
            if use_user_ids:
                self.data_used.append('user id')
            if use_urls:
                self.data_used.append('url')

        self.start_timestamp = None
        self.end_timestamp = None
        self.data = pd.DataFrame(columns=self.data_used)
        self.processor = Processor()

        if w2v_file:
            wv = KeyedVectors.load_word2vec_format(w2v_file)
            w2v = dict(zip(list(wv.vocab.keys()), wv.vectors))
            config.WORD_2_VEC = w2v
        if pos:
            config.POS = pos

        if not config.POS_THRESHOLD:
            config.POS_THRESHOLD = pos_threshold
        if not config.NUM_CLASSES:
            config.NUM_CLASSES = num_classes
        if not config.NEG_THRESHOLD:
            config.NEG_THRESHOLD = neg_threshold
        if not config.RATING_TYPE:
            config.RATING_TYPE = rating_type
        if not config.DATA_REPRESENTATION:
            config.DATA_REPRESENTATION = data_representation

    def collect(self, url):
        """
        Given a url, collects drug review data into Dataset
        :param url: drug reviews url
        """
        assert self.scraper, "In order to collect reviews, a scraper must be specified"

        if not self.start_timestamp:
            self.start_timestamp = str(datetime.datetime.now())

        self.scraper.scrape(url)
        self.data = self.data.append(self.scraper.dataset, ignore_index=True)

        self.end_timestamp = str(datetime.datetime.now())

    def collect_from_drug_names(self, drug_names_file, output_file, start=0):
        """
        Given a text file listing drug names, collects a dataset of reviews for those drugs
        :param drug_names_file: path to urls file
        :param output_file: where to output data
        :param start: from where in the urls file to start scraping
        """
        assert self.scraper, "In order to collect reviews, a scraper must be specified"

        if not self.start_timestamp:
            self.start_timestamp = str(datetime.datetime.now())

        print('\nCollecting urls...')
        self.scraper.get_urls(drug_names_file, 'medinify/scrapers/temp_urls_file.txt')
        print('\nScraping urls...')
        self.scraper.scrape_urls('medinify/scrapers/temp_urls_file.txt', output_file=output_file, start=start)
        self.data = self.data.append(self.scraper.dataset, ignore_index=True)

        os.remove('medinify/scrapers/temp_urls_file.txt')
        print('Collected reviews.')

        self.end_timestamp = str(datetime.datetime.now())

    def collect_from_urls(self, urls_file, output_file, start=0):
        """
        Given a file listing drug urls, collects review data into Dataset
        :param urls_file: path to file listing drug urls
        :param output_file: where to output data
        :param start: from where in the urls file to start scraping
        """
        assert self.scraper, "In order to collect reviews, a scraper must be specified"

        if not self.start_timestamp:
            self.start_timestamp = str(datetime.datetime.now())

        print('\nScraping urls...')
        self.scraper.scrape_urls(urls_file, output_file=output_file, start=start)
        self.data = self.data.append(self.scraper.dataset, ignore_index=True)

        self.end_timestamp = str(datetime.datetime.now())

    def write_file(self, output_file, write_comments=True,
                   write_ratings=True, write_date=True,
                   write_drugs=True, write_user_ids=False,
                   write_urls=False):
        """
        Write csv file containing data
        :param output_file: csv output file path
        :param write_comments: whether or not to write comments to csv file
        :param write_ratings: whether or not to write ratings to csv file
        :param write_date: whether or not to write dates to csv file
        :param write_drugs: whether or not to write drug names to csv file
        :param write_urls: whether or not to write urls to csv file
        :param write_user_ids: whether or not to write urls to csv file
        """
        columns = []
        if write_comments:
            columns.append('comment')
        if write_ratings:
            columns.append('rating')
        if write_date:
            columns.append('date')
        if write_drugs:
            columns.append('drug')
        if write_user_ids:
            columns.append('user id')
        if write_urls:
            columns.append('url')
        self.remove_empty_comments()
        self.remove_duplicate_comments()
        self.remove_float_comments()
        self.data.to_csv(output_file, columns=columns, index=False)

    def save_data(self, output_file):
        """
        Saves Dataset in compressed pickle file
        :param output_file: path to output pickle file
        """
        with open(output_file, 'wb') as pkl:
            pickle.dump(self.data, pkl, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.start_timestamp, pkl, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.end_timestamp, pkl, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.processor, pkl, protocol=pickle.HIGHEST_PROTOCOL)

    def load_data(self, pickle_file):
        """
        Loads dataset from compressed pickle file
        :param pickle_file: path to saved pickle file
        """
        with open(pickle_file, 'rb') as pkl:
            self.data = pickle.load(pkl)
            self.start_timestamp = pickle.load(pkl)
            self.end_timestamp = pickle.load(pkl)
            self.processor = pickle.load(pkl)
        self.remove_float_comments()

    def load_file(self, csv_file):
        """
        Loads dataset from csv file
        :param csv_file: path to csv file to load
        """
        self.data = pd.read_csv(csv_file)

    def remove_empty_comments(self):
        """
        Removes empty comments from Dataset
        """
        data_array = self.data.to_numpy()
        new_list = []
        num_empty = 0

        for review in data_array:
            if type(review[0]) != float:
                new_list.append(review)
            else:
                num_empty += 1

        new_array = np.asarray(new_list)
        self.data = pd.DataFrame(new_array, columns=self.data_used)
        print('Removed {} empty comment(s).'.format(num_empty))

    def remove_duplicate_comments(self):
        """
        Removes duplicate comments from Dataset
        """
        data_array = self.data.to_numpy()
        not_dupes = []
        not_dupes_reviews = []
        num_dupes = 0

        for review in data_array:
            if review[0] not in not_dupes_reviews:
                not_dupes_reviews.append(review[0])
                not_dupes.append(review)
            else:
                num_dupes += 1

        new_array = np.asarray(not_dupes)
        self.data = pd.DataFrame(new_array, columns=self.data_used)
        print('Removed {} duplicate comment(s).'.format(num_dupes))

    def remove_float_comments(self):
        """
        Removes float comments from Dataset
        """
        data_array = self.data.to_numpy()
        not_floats = []
        num_floats = 0

        for review in data_array:
            if type(review[0]) == str:
                not_floats.append(review)
            else:
                num_floats += 1

        new_array = np.asarray(not_floats)
        self.data = pd.DataFrame(new_array, columns=self.data_used)
        print('Removed {} float comment(s).'.format(num_floats))

    def print_stats(self):
        """
        Calculates and prints data distribution statistics
        """
        ratings = self.data['rating'].to_numpy()
        num_reviews = len(ratings)

        if type(ratings[0]) == str and ratings[0][0] == '{':
            ratings = [ast.literal_eval(x) for x in ratings]
            rating_types = list(ratings[0].keys())
            ratings_sets = {rating_type: [] for rating_type in rating_types}
            data = {rating_type: {} for rating_type in rating_types}
            for rating in ratings:
                for rating_type in rating_types:
                    ratings_sets[rating_type].append(float(rating[rating_type]))
            for rating_type in rating_types:
                ratings_sets[rating_type] = np.asarray(ratings_sets[rating_type])
                data[rating_type]['range'] = (np.amin(ratings_sets[rating_type]),
                                              np.amax(ratings_sets[rating_type]))
                data[rating_type]['num_pos'] = len([x for x in ratings_sets[rating_type]
                                                    if x >= config.POS_THRESHOLD])
                data[rating_type]['num_neg'] = len([x for x in ratings_sets[rating_type]
                                                    if x <= config.NEG_THRESHOLD])
                data[rating_type]['num_neutral'] = len([x for x in ratings_sets[rating_type]
                                                        if config.NEG_THRESHOLD < x < config.POS_THRESHOLD])

            print('\nDataset Stats:\n')
            print('Number of reviews with ratings: {}'.format(num_reviews))
            print('Types of ratings: {}'.format(rating_types))
            print('\nRating type distributions:\n')
            for rating_type in rating_types:
                print('{}:'.format(rating_type))
                print('\tRating Range: {}'.format(data[rating_type]['range']))
                print('\tPositive Reviews: {}'.format(data[rating_type]['num_pos']))
                print('\tNegative Reviews: {}'.format(data[rating_type]['num_neg']))
                print('\tNeutral Reviews: {}'.format(data[rating_type]['num_neutral']))
                print('\tPos:Neg Ratio: {}\n'.format(data[rating_type]['num_pos'] / data[rating_type]['num_neg']))

        elif type(ratings[0]) == np.float64:
            ratings = np.asarray([x for x in ratings if not np.isnan(x)])
            num_reviews = len(ratings)
            range_ = (np.amin(ratings), np.amax(ratings))
            num_pos = len([x for x in ratings if x >= config.POS_THRESHOLD])
            num_neg = len([x for x in ratings if x <= config.NEG_THRESHOLD])
            num_neutral = len([x for x in ratings if config.NEG_THRESHOLD < x < config.POS_THRESHOLD])

            print('\nDataset Stats:\n')
            print('Number of reviews with ratings: {}'.format(num_reviews))
            print('Rating Range: {}'.format(range_))
            print('Positive Reviews: {}'.format(num_pos))
            print('Negative Reviews: {}'.format(num_neg))
            print('Neutral Reviews: {}'.format(num_neutral))
            print('Pos:Neg Ratio: {}\n'.format(num_pos / num_neg))

        else:
            raise ValueError('This type of rating ({}) is not supported.'.format(type(ratings[0])))

    def get_count_vectors(self, classifying=False):
        """
        Wraps around Processor function
        :param classifying: if running classification on data
        :return: data, target
        """
        self.remove_empty_comments()
        self.remove_duplicate_comments()
        self.remove_float_comments()

        reviews = self.processor.get_count_vectors(self.data['comment'], self.data['rating'])
        data, target, comments = [], [], []
        for review in reviews:
            if config.NUM_CLASSES == 2 and review.target in [0.0, 1.0]:
                data.append(review.data)
                target.append(review.target)
                comments.append(review.comment)
            elif config.NUM_CLASSES == 3 and review.target in [0.0, 1.0, 2.0]:
                data.append(review.data)
                target.append(review.target)
                comments.append(review.comment)
            elif config.NUM_CLASSES == 5 and review.target in [0.0, 1.0, 2.0, 3.0, 4.0]:
                data.append(review.data)
                target.append(review.target)
                comments.append(review.comment)

        if classifying:
            return data, target, comments
        else:
            return data, target

    def get_tfidf_vectors(self, classifying=False):
        """
        Wraps around Processor function
        :param classifying: if running classification on data
        :return: data, target
        """
        self.remove_empty_comments()
        self.remove_duplicate_comments()
        self.remove_float_comments()

        reviews = self.processor.get_tfidf_vectors(self.data['comment'], self.data['rating'])
        data, target, comments = [], [], []
        for review in reviews:
            if config.NUM_CLASSES == 2 and review.target in [0.0, 1.0]:
                data.append(review.data)
                target.append(review.target)
                comments.append(review.comment)
            elif config.NUM_CLASSES == 3 and review.target in [0.0, 1.0, 2.0]:
                data.append(review.data)
                target.append(review.target)
                comments.append(review.comment)
            elif config.NUM_CLASSES == 5 and review.target in [0.0, 1.0, 2.0, 3.0, 4.0]:
                data.append(review.data)
                target.append(review.target)
                comments.append(review.comment)

        if classifying:
            return data, target, comments
        else:
            return data, target

    def get_average_embeddings(self, classifying=False):
        """
        Wraps around Processor function
        :param classifying: if running classification on data
        :return: data, target
        """
        self.remove_empty_comments()
        self.remove_duplicate_comments()
        self.remove_float_comments()

        reviews = self.processor.get_average_embeddings(self.data['comment'], self.data['rating'])
        data, target, comments = [], [], []
        for review in reviews:
            if not np.sum(review.data) == 0:
                if config.NUM_CLASSES == 2 and review.target in [0.0, 1.0]:
                    data.append(review.data)
                    target.append(review.target)
                    comments.append(review.comment)
                elif config.NUM_CLASSES == 3 and review.target in [0.0, 1.0, 2.0]:
                    data.append(review.data)
                    target.append(review.target)
                    comments.append(review.comment)
                elif config.NUM_CLASSES == 5 and review.target in [0.0, 1.0, 2.0, 3.0, 4.0]:
                    data.append(review.data)
                    target.append(review.target)
                    comments.append(review.comment)

        if classifying:
            return data, target, comments
        else:
            return data, target

    def get_pos_vectors(self, classifying=False):
        """
        Wraps around Processor function
        :param classifying: if running classification on data
        :return: data, target
        """
        self.remove_empty_comments()
        self.remove_duplicate_comments()
        self.remove_float_comments()

        reviews = self.processor.get_pos_vectors(self.data['comment'], self.data['rating'])
        data, target, comments = [], [], []
        for review in reviews:
            if not np.sum(review.data) == 0:
                if config.NUM_CLASSES == 2 and review.target in [0.0, 1.0]:
                    data.append(review.data)
                    target.append(review.target)
                    comments.append(review.comment)
                elif config.NUM_CLASSES == 3 and review.target in [0.0, 1.0, 2.0]:
                    data.append(review.data)
                    target.append(review.target)
                    comments.append(review.comment)
                elif config.NUM_CLASSES == 5 and review.target in [0.0, 1.0, 2.0, 3.0, 4.0]:
                    data.append(review.data)
                    target.append(review.target)
                    comments.append(review.comment)

        if classifying:
            return data, target, comments
        else:
            return data, target



