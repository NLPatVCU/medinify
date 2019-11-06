
from medinify.datasets import Dataset
from medinify import scrapers
import pandas as pd
import os
import ast
import numpy as np


class SentimentDataset(Dataset):
    """
    SentimentDataset is a subclass of Dataset, and provides all the same loading,
    storing, cleaning, and writing functionality
    Also provides functionality for scraping drug review sentiment analysis datasets
    from WebMD.com, Drugs.com DrugRatingz.com, and EverydayHealth.com
    (SentimentDataset was split off to make Dataset a more generalized class)

    Attributes:
        text_column:    (str) Column name from data csv for text data
        label_column: (str) Column name from data csv for label data
        data_table:         (pandas DataFrame) Where all data is internally stored
        num_classes: (int) number of star rating classes to use when generating labels (2 for polarity classification,
            3 or 5 for more specific degrees of sentiment)
        scraper: (Scraper) scraper to use when collecting datasets
    """
    def __init__(self, csv_file=None, text_column='comment', label_column='effectiveness', scraper='webmd',
                 collect_user_ids=False, collect_urls=False, num_classes=2):
        """
        Constructor for SentimentDataset
        Sets up what data will be processed as text and label, and loads data
        into DataFrame if csv path is provided, then generates sentiment labels based on num_classes
        :param csv_file: (str) Path to csv file with stored data
        :param text_column: (str) name of the csv column containing the text data
        :param label_column: (str) name of the csv column containing the label data
        :param scraper: (Scraper) scraper to use when collecting datasets
        :param collect_user_ids: (boolean) whether or not not collect user ids when scraping
        :param collect_urls: (boolean) whether or not to collect urls when scraping
        :param num_classes: (int) number of star rating classes to use when generating labels
        """
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
        """
        Given a url, scrapes, stores, cleans, and generates labels for review data
        :param url: (str) url for drug reviews page to scraper
        """
        self.scraper.scrape(url)
        scraped_data = pd.DataFrame(self.scraper.reviews)
        self.data_table = self.data_table.append(scraped_data, ignore_index=True)
        self.transform_old_dataset()
        self._clean_data()
        self.generate_labels()

    def collect_from_urls(self, urls_file=None, urls=None, start=0):
        """
        Collect drug review data from list of urls
        :param urls_file: (str) path to file containing list of drug review urls
        :param urls: (list[str]) list of drug review urls
        :param start: (int) where to start in list of urls (if restarting scraping)
        """
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
        """
        Collect drug review data from list of drug names
        :param drug_names_file: (str) path to file containing list of drug names
        :param start: (int) where to start in list of urls (if restarting scraping)
        """
        print('\nCollecting urls...')
        urls = self.scraper.get_urls(drug_names_file)
        self.collect_from_urls(urls=urls, start=start)

    def write_file(self, output_file):
        """
        Writes file of the current internal data (data_table)
        Searches for data/csvs directory, saves csv there (also transforms data
        from old, scraper structure to new structure before writing file)
        :param output_file: (str) name to save file to (should end with .csv)
        """
        if 'ratings' in list(self.data_table.columns.values):
            self.transform_old_dataset()
        super().write_file(output_file)

    def load_file(self, csv_file):
        """
        Loads a csv file's data into Sentiment Dataset's internal storage
        Removes empty elements with empty text or empty label (transforms old format if needed)
        :param csv_file: (str) path to csv file with data
        """
        super().load_file(csv_file)
        if 'rating' in list(self.data_table.columns.values):
            self.transform_old_dataset()
        self.generate_labels()

    def transform_old_dataset(self):
        """
        Transforms old rating format (one column containing a dictionary) into new
        format (one column per rating type)
        """
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
        """
        Generates sentiment labels from star ratings, stores them in new column
        """
        if 'label' not in list(self.data_table.columns.values):
            labels = self.data_table[self.label_column].apply(lambda x: self._rating_to_label(x))
            self.data_table['label'] = labels
            self.label_column = 'label'
            if self.num_classes == 2:
                self.data_table = self.data_table.loc[self.data_table['label'].notnull()]

    def _rating_to_label(self, rating):
        """
        Transforms star ratings into sentiment labels based on num_classes
        :param rating: (int) rating to transform
        :return: sentiment label
        """
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

