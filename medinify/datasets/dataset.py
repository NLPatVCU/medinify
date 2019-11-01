
import pandas as pd
import os
import ast
from medinify.scrapers.webmd_scraper import WebMDScraper
from medinify.scrapers.drugs_scraper import DrugsScraper
from medinify.scrapers.drugratingz_scraper import DrugRatingzScraper
from medinify.scrapers.everydayhealth_scraper import EverydayHealthScraper


class Dataset:

    def __init__(self, scraper='WebMD', collect_user_ids=False, collect_urls=False,
                 num_classes=2, text_column='comment', label_column='effectiveness',
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
        self.text_column = text_column
        self.label_column = label_column
        self.word_embeddings = word_embeddings
        self.feature_representation = feature_representation

    def collect(self, url):
        self.scraper.scrape(url)
        scraped_data = pd.DataFrame(self.scraper.reviews)
        self.data_table = self.data_table.append(scraped_data, ignore_index=True)
        self._clean_comments()
        self.transform_old_dataset()

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
        self.transform_old_dataset()

    def collect_from_drug_names(self, drug_names_file, start=0):
        print('\nCollecting urls...')
        urls = self.scraper.get_urls(drug_names_file)
        self.collect_from_urls(urls=urls, start=start)

    def write_file(self, output_file):
        if 'ratings' in list(self.data_table.columns.values):
            self.transform_old_dataset()
        self.data_table.to_csv('./data/' + output_file, index=False)

    def load_file(self, csv_file):
        self.data_table = pd.read_csv('./data/' + csv_file)
        self._clean_comments()
        if 'rating' in list(self.data_table.columns.values):
            self.transform_old_dataset()

    def _remove_empty_comments(self):
        num_rows = len(self.data_table)
        self.data_table = self.data_table.loc[self.data_table[self.text_column].notnull()]
        self.data_table = self.data_table.loc[self.data_table[self.text_column] != '']
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
        labels = self.data_table[self.label_column]
        print('\n******************************************************************************************\n')
        print('Dataset Stats:\n')
        print('Total reviews: %d' % len(labels))
        unique_labels = set(labels)
        print('Number of Unique Labels: %d\t(%s)\n' % (
            len(unique_labels), ', '.join([str(x) for x in unique_labels])))

        print('Label Stats:')
        for label in unique_labels:
            num_label = labels.loc[labels == label].shape[0]
            percent = 100 * (num_label / len(labels))
            print('\tLabel: %s\t\tNumber of Instance: %d\t\tPercent of Instances: %.2f%%' % (
                str(label), num_label, percent))
        print('\n******************************************************************************************\n')

    def transform_old_dataset(self):
        if type(self.data_table.iloc[0]['rating']) == str:
            new_columns = list(ast.literal_eval(self.data_table.iloc[0]['rating']).keys())
        else:
            new_columns = list(self.data_table.iloc[0]['rating'].keys())
        for column in new_columns:
            if type(self.data_table.iloc[0]['rating']) == str:
                self.data_table[column] = self.data_table.apply(
                    lambda row: ast.literal_eval(row['rating'])[column], axis=1)
            else:
                self.data_table[column] = self.data_table.apply(
                    lambda row: row['rating'][column], axis=1)
        self.data_table.drop(['rating'], axis=1, inplace=True)




