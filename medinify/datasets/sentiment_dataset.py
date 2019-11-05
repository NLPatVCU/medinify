
from medinify.datasets import Dataset
from medinify import scrapers
import pandas as pd
import os
import ast
import numpy as np


class SentimentDataset(Dataset):

    def __init__(self, csv_file=None, text_column='comment', label_column='effectiveness', scraper='webmd',
                 collect_user_ids=False, collect_urls=False, num_classes=2):
        self.num_classes = num_classes
        super().__init__(csv_file=csv_file, text_column=text_column, label_column=label_column)
        for scrap in scrapers.Scraper.__subclasses__():
            if scrap.nickname == scraper:
                self.scraper = scrap(collect_urls=collect_urls, collect_user_ids=collect_user_ids)

        if type(self.data_table) != pd.DataFrame:
            columns = ['comment', 'rating', 'date', 'drug']
            if collect_urls:
                columns.append('url')
            if collect_user_ids:
                columns.append('user id')

            self.data_table = pd.DataFrame(columns=columns)
        else:
            self.generate_labels()

    def collect(self, url):
        self.scraper.scrape(url)
        scraped_data = pd.DataFrame(self.scraper.reviews)
        self.data_table = self.data_table.append(scraped_data, ignore_index=True)
        self.transform_old_dataset()
        self._clean_data()
        self.generate_labels()

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
        super().write_file(output_file)

    def load_file(self, csv_file):
        super().load_file(csv_file)
        if 'rating' in list(self.data_table.columns.values):
            self.transform_old_dataset()
        self.generate_labels()

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

    def generate_labels(self):
        if 'label' not in list(self.data_table.columns.values):
            labels = self.data_table[self.label_column].apply(lambda x: self._rating_to_label(x))
            self.data_table['label'] = labels
            self.label_column = 'label'
            if self.num_classes == 2:
                self.data_table = self.data_table.loc[self.data_table['label'].notnull()]

    def _rating_to_label(self, rating):
        if self.num_classes == 2:
            if rating in [1.0, 2.0]:
                return 0
            elif rating in [4.0, 5.0]:
                return 1
            else:
                return np.NaN
        elif self.num_classes == 3:
            if rating in [1.0, 2.0]:
                return 0
            elif rating in [4.0, 5.0]:
                return 2
            else:
                return 1

