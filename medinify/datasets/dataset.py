
import pandas as pd
import os
import ast
from gensim.models import KeyedVectors
from medinify.scrapers.webmd_scraper import WebMDScraper
from medinify.scrapers.drugs_scraper import DrugsScraper
from medinify.scrapers.drugratingz_scraper import DrugRatingzScraper
from medinify.scrapers.everydayhealth_scraper import EverydayHealthScraper
from medinify.datasets.process.process import Processor


class Dataset:

    def __init__(self, scraper='WebMD', collect_user_ids=False, collect_urls=False, word_embeddings=None):
        assert scraper in ['WebMD', 'Drugs', 'DrugRatingz', 'EverydayHealth'], \
            'Scraper must be \'WebMD\', \'Drugs\', \'DrugRatingz\', or \'EverydayHealth\''
        if scraper == 'WebMD':
            self.scraper = WebMDScraper(collect_user_ids=collect_user_ids, collect_urls=collect_urls)
        elif scraper == 'Drugs':
            self.scraper = DrugsScraper(collect_user_ids=collect_user_ids, collect_urls=collect_urls)
        elif scraper == 'DrugRatingz':
            if collect_user_ids:
                raise AttributeError('DrugRatingz scraper cannot collect user ids')
            self.scraper = DrugRatingzScraper(collect_urls=collect_urls)
        elif scraper == 'EverydayHealth':
            if collect_user_ids:
                raise AttributeError('EverydayHealth scraper cannot collect user ids')
            self.scraper = EverydayHealthScraper(collect_urls=collect_urls)

        columns = ['comment', 'rating', 'date', 'drug']
        if collect_urls:
            columns.append('url')
        if collect_user_ids:
            columns.append('user id')

        self.dataset = pd.DataFrame(columns=columns)
        self.processor = Processor()
        self.count_vectors = None
        self.average_embeddings = None
        self.word_embeddings_file = word_embeddings

    def collect(self, url):
        self.scraper.scrape(url)
        scraped_data = pd.DataFrame(self.scraper.reviews)
        self.dataset = self.dataset.append(scraped_data, ignore_index=True)
        self._clean_comments()

    def collect_from_urls(self, urls_file=None, urls=None, start=0):
        assert bool(urls_file) ^ bool(urls)
        if urls_file:
            with open(urls_file, 'r') as f:
                urls = [x[:-1] for x in f.readlines()]
        if start != 0:
            if os.path.exists('./medinify/datasets/temp_file.csv'):
                self.dataset = pd.read_csv('./medinify/datasets/temp_file.csv')
            else:
                print('No saved data found for urls 0 - %d' % start)
        print('\nScraping urls...')
        for url in urls[start:]:
            self.collect(url)
            self.dataset.to_csv('./medinify/datasets/temp_file.csv', index=False)
            start += 1

            print('\nTemporary review data file saved.')
            print('Safe to quit. Start from %d.' % start)
        print('Finished collection.')
        os.remove('./medinify/datasets/temp_file.csv')

    def collect_from_drug_names(self, drug_names_file, start=0):
        print('\nCollecting urls...')
        urls = self.scraper.get_urls(drug_names_file)
        self.collect_from_urls(urls=urls, start=start)

    def write_file(self, output_file):
        self.dataset.to_csv(output_file, index=False)

    def load_file(self, csv_file):
        self.dataset = pd.read_csv(csv_file)
        self._clean_comments()

    def _remove_empty_comments(self):
        num_rows = len(self.dataset)
        self.dataset = self.dataset.loc[self.dataset['comment'].notnull()]
        num_removed = num_rows - len(self.dataset)
        print('Removed %d empty comment(s).' % num_removed)

    def _remove_duplicate_comments(self):
        num_rows = len(self.dataset)
        self.dataset.drop_duplicates(subset='comment', inplace=True)
        num_removed = num_rows - len(self.dataset)
        print('Removed %d duplicate review(s).' % num_removed)

    def _clean_comments(self):
        self._remove_duplicate_comments()
        self._remove_empty_comments()

    def print_stats(self):
        if type(self.scraper) != WebMDScraper:
            raise NotImplementedError('print_stats function is not implemented for non-WebMD dataset.')
        print('\n***********************************************\n')
        print('Dataset Stats:\n')
        print('Total reviews: %d' % len(self.dataset))
        pos_effectiveness, pos_satisfaction, pos_ease_of_use = 0, 0, 0
        neg_effectiveness, neg_satisfaction, neg_ease_of_use = 0, 0, 0
        neut_effectiveness, neut_satisfaction, neut_ease_of_use = 0, 0, 0
        for rating in self.dataset['rating']:
            ratings = ast.literal_eval(rating)
            if ratings['effectiveness'] in [1.0, 2.0]:
                neg_effectiveness += 1
            elif ratings['effectiveness'] in [4.0, 5.0]:
                pos_effectiveness += 1
            else:
                neut_effectiveness += 1
            if ratings['ease of use'] in [1.0, 2.0]:
                neg_ease_of_use += 1
            elif ratings['ease of use'] in [4.0, 5.0]:
                pos_ease_of_use += 1
            else:
                neut_ease_of_use += 1
            if ratings['satisfaction'] in [1.0, 2.0]:
                neg_satisfaction += 1
            elif ratings['satisfaction'] in [4.0, 5.0]:
                pos_satisfaction += 1
            else:
                neut_satisfaction += 1
        print('\nEffectiveness Ratings:')
        print('\tPositive: %d (%.2f%%)\n\tNegative: %d (%.2f%%)\n\tNeutral: %d (%.2f%%)' %
              (pos_effectiveness, 100 * (pos_effectiveness / len(self.dataset)),
               neg_effectiveness, 100 * (neg_effectiveness / len(self.dataset)),
               neut_effectiveness, 100 * (neut_effectiveness / len(self.dataset))))
        print('\nSatisfaction Ratings:')
        print('\tPositive: %d (%.2f%%)\n\tNegative: %d (%.2f%%)\n\tNeutral: %d (%.2f%%)' %
              (pos_satisfaction, 100 * (pos_satisfaction / len(self.dataset)),
               neg_satisfaction, 100 * (neg_satisfaction / len(self.dataset)),
               neut_satisfaction, 100 * (neut_satisfaction / len(self.dataset))))
        print('\nEase of Use Ratings:')
        print('\tPositive: %d (%.2f%%)\n\tNegative: %d (%.2f%%)\n\tNeutral: %d (%.2f%%)' %
              (pos_ease_of_use, 100 * (pos_ease_of_use / len(self.dataset)),
               neg_ease_of_use, 100 * (neg_ease_of_use / len(self.dataset)),
               neut_ease_of_use, 100 * (neut_ease_of_use / len(self.dataset))))
        print('\n***********************************************\n')

    def get_count_vectors(self):
        self.count_vectors = self.processor.process_count_vectors(self.dataset['comment'])

    def get_average_embeddings(self):
        w2v = KeyedVectors.load_word2vec_format(self.word_embeddings_file)
        self.average_embeddings = self.processor.get_average_embeddings(self.dataset['comment'], w2v)

    """
    def get_pos_vectors(self, classifying=False):
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
    """



