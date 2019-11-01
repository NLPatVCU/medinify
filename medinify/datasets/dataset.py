
import pandas as pd
import os
import ast
from medinify.scrapers.webmd_scraper import WebMDScraper
from medinify.scrapers.drugs_scraper import DrugsScraper
from medinify.scrapers.drugratingz_scraper import DrugRatingzScraper
from medinify.scrapers.everydayhealth_scraper import EverydayHealthScraper


class Dataset:

    def __init__(self, scraper='WebMD', collect_user_ids=False, collect_urls=False,
                 num_classes=2, text_column='comment', feature_column='effectiveness',
                 word_embeddings=None, feature_representation='bow'):
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
        self.data_table = pd.DataFrame(columns=columns)
        self.num_classes = num_classes
        self.feature_column = feature_column
        self.text_column = text_column
        self.word_embeddings = word_embeddings
        self.feature_representation = feature_representation

    def collect(self, url):
        self.scraper.scrape(url)
        scraped_data = pd.DataFrame(self.scraper.reviews)
        self.data_table = self.data_table.append(scraped_data, ignore_index=True)
        self._clean_comments()
        self.data_table = self.transform_old_dataset(data=self.data_table)

    def collect_from_urls(self, urls_file=None, urls=None, start=0):
        assert bool(urls_file) ^ bool(urls)
        if urls_file:
            with open(urls_file, 'r') as f:
                urls = [x[:-1] for x in f.readlines()]
        if start != 0:
            if os.path.exists('./medinify/datasets/temp_file.csv'):
                self.data_table = pd.read_csv('./medinify/datasets/temp_file.csv')
            else:
                print('No saved data found for urls 0 - %d' % start)
        print('\nScraping urls...')
        for url in urls[start:]:
            self.collect(url)
            self.data_table.to_csv('./medinify/datasets/temp_file.csv', index=False)
            start += 1

            print('\nTemporary review data file saved.')
            print('Safe to quit. Start from %d.' % start)
        print('Finished collection.')
        os.remove('./medinify/datasets/temp_file.csv')
        self.data_table = self.transform_old_dataset(data=self.data_table)

    def collect_from_drug_names(self, drug_names_file, start=0):
        print('\nCollecting urls...')
        urls = self.scraper.get_urls(drug_names_file)
        self.collect_from_urls(urls=urls, start=start)

    def write_file(self, output_file):
        if 'ratings' in list(self.data_table.columns.values):
            self.data_table = self.transform_old_dataset(data=self.data_table)
        self._clean_comments()
        self.data_table.to_csv(output_file, index=False)

    def load_file(self, csv_file):
        self.data_table = pd.read_csv(csv_file)
        self._clean_comments()
        if 'rating' in list(self.data_table.columns.values):
            self.data_table = self.transform_old_dataset(data=self.data_table)

    def _remove_empty_comments(self):
        num_rows = len(self.data_table)
        self.data_table = self.data_table.loc[self.data_table[self.text_column].notnull()]
        num_removed = num_rows - len(self.data_table)
        print('Removed %d empty comment(s).' % num_removed)

    def _remove_duplicate_comments(self):
        num_rows = len(self.data_table)
        self.data_table.drop_duplicates(subset='comment', inplace=True)
        num_removed = num_rows - len(self.data_table)
        print('Removed %d duplicate review(s).' % num_removed)

    def _clean_comments(self):
        self._remove_duplicate_comments()
        self._remove_empty_comments()

    def print_stats(self):
        if type(self.scraper) != WebMDScraper:
            raise NotImplementedError('print_stats function is not implemented for non-WebMD dataset.')
        print('\n***********************************************\n')
        print('Dataset Stats:\n')
        print('Total reviews: %d' % len(self.data_table))
        if 'ratings' in list(self.data_table.columns.values):
            self.data_table = self.transform_old_dataset(data=self.data_table)

        pos_effectiveness = self.data_table.loc[self.data_table['effectiveness'] == 4.0].shape[0] + \
                            self.data_table.loc[self.data_table['effectiveness'] == 5.0].shape[0]
        neut_effectiveness = self.data_table.loc[self.data_table['effectiveness'] == 3.0].shape[0]
        neg_effectiveness = self.data_table.loc[self.data_table['effectiveness'] == 1.0].shape[0] + \
                            self.data_table.loc[self.data_table['effectiveness'] == 2.0].shape[0]

        pos_satisfaction = self.data_table.loc[self.data_table['satisfaction'] == 4.0].shape[0] + \
                           self.data_table.loc[self.data_table['satisfaction'] == 5.0].shape[0]
        neut_satisfaction = self.data_table.loc[self.data_table['satisfaction'] == 3.0].shape[0]
        neg_satisfaction = self.data_table.loc[self.data_table['satisfaction'] == 1.0].shape[0] + \
                           self.data_table.loc[self.data_table['satisfaction'] == 2.0].shape[0]

        pos_ease_of_use = self.data_table.loc[self.data_table['ease of use'] == 4.0].shape[0] + \
                          self.data_table.loc[self.data_table['ease of use'] == 5.0].shape[0]
        neut_ease_of_use = self.data_table.loc[self.data_table['ease of use'] == 3.0].shape[0]
        neg_ease_of_use = self.data_table.loc[self.data_table['ease of use'] == 1.0].shape[0] + \
                          self.data_table.loc[self.data_table['ease of use'] == 2.0].shape[0]

        print('\nEffectiveness Ratings:')
        print('\tPositive: %d (%.2f%%)\n\tNegative: %d (%.2f%%)\n\tNeutral: %d (%.2f%%)' %
              (pos_effectiveness, 100 * (pos_effectiveness / len(self.data_table)),
               neg_effectiveness, 100 * (neg_effectiveness / len(self.data_table)),
               neut_effectiveness, 100 * (neut_effectiveness / len(self.data_table))))
        print('\nSatisfaction Ratings:')
        print('\tPositive: %d (%.2f%%)\n\tNegative: %d (%.2f%%)\n\tNeutral: %d (%.2f%%)' %
              (pos_satisfaction, 100 * (pos_satisfaction / len(self.data_table)),
               neg_satisfaction, 100 * (neg_satisfaction / len(self.data_table)),
               neut_satisfaction, 100 * (neut_satisfaction / len(self.data_table))))
        print('\nEase of Use Ratings:')
        print('\tPositive: %d (%.2f%%)\n\tNegative: %d (%.2f%%)\n\tNeutral: %d (%.2f%%)' %
              (pos_ease_of_use, 100 * (pos_ease_of_use / len(self.data_table)),
               neg_ease_of_use, 100 * (neg_ease_of_use / len(self.data_table)),
               neut_ease_of_use, 100 * (neut_ease_of_use / len(self.data_table))))
        print('\n***********************************************\n')

    @staticmethod
    def transform_old_dataset(csv_file=None, data=None, output_file=None):
        if csv_file:
            data = pd.read_csv(csv_file)
        if type(data.iloc[0]['rating']) == str:
            new_columns = list(ast.literal_eval(data.iloc[0]['rating']).keys())
        else:
            new_columns = list(data.iloc[0]['rating'].keys())
        for column in new_columns:
            if type(data.iloc[0]['rating']) == str:
                data[column] = data.apply(lambda row: ast.literal_eval(row['rating'])[column], axis=1)
            else:
                data[column] = data.apply(lambda row: row['rating'][column], axis=1)
        data.drop(['rating'], axis=1, inplace=True)
        if output_file:
            data.to_csv(output_file, index=False)
        return data




