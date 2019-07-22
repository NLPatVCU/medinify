
import pandas as pd
from medinify.scrapers.webmd_scraper import WebMDScraper
from medinify.scrapers.drugs_scraper import DrugsScraper
from medinify.scrapers.drugratingz_scraper import DrugRatingzScraper
from medinify.scrapers.everydayhealth_scraper import EverydayHealthScraper
import os


class Dataset:

    data_used = []
    data = None
    scraper = None

    def __init__(self, scraper=None, use_rating=True,
                 use_dates=True, use_drugs=True,
                 use_user_ids=False, use_urls=False):
        """
        Creates an instance of the Dataset class, which stores and processes data
        If also wraps around the functionality of the review scrapers for collecting review data
        :param scraper: Which scraper to use for scraping functionality
        :param use_rating: whether or not to store rating data
        :param use_dates: whether or not to store date data
        :param use_drugs: whether or not to store drug name data
        :param use_user_ids: whether or not to store user id data
        :param use_urls: whether or not to store drug url data
        """
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

        if scraper == 'WebMD':
            self.scraper = WebMDScraper(collect_ratings=use_rating, collect_dates=use_dates,
                                        collect_drugs=use_drugs, collect_user_ids=use_user_ids,
                                        collect_urls=use_urls)
        if scraper == 'Drugs':
            self.scraper = DrugsScraper(collect_ratings=use_rating, collect_dates=use_dates,
                                        collect_drugs=use_drugs, collect_user_ids=use_user_ids,
                                        collect_urls=use_urls)
        if scraper == 'DrugRatingz':
            self.scraper = DrugRatingzScraper(collect_ratings=use_rating, collect_dates=use_dates,
                                              collect_drugs=use_drugs, collect_user_ids=use_user_ids,
                                              collect_urls=use_urls)
        if scraper == 'EverydayHealth':
            self.scraper = EverydayHealthScraper(collect_ratings=use_rating, collect_dates=use_dates,
                                                 collect_drugs=use_drugs, collect_user_ids=use_user_ids,
                                                 collect_urls=use_urls)

        self.data = pd.DataFrame(columns=self.data_used)

    def add_reviews(self, reviews):
        """
        Adds a dataframe of reviews to dataset
        :param reviews: DataFrame of reviews data
        """
        self.data.append(reviews, ignore_index=True)

    def collect(self, url):
        """
        Given a url, collects drug review data into Dataset
        :param url: drug reviews url
        """
        assert self.scraper, "In order to collect reviews, a scraper must be specified"

        self.scraper.scrape(url)
        self.data = self.data.append(self.scraper.dataset, ignore_index=True)

    def collect_from_drug_names(self, drug_names_file):
        """
        Given a text file listing drug names, collects a dataset of reviews for those drugs
        :param drug_names_file: path to urls file
        """
        assert self.scraper, "In order to collect reviews, a scraper must be specified"

        print('\nCollecting urls...')
        self.scraper.get_urls(drug_names_file, 'medinify/scrapers/temp_urls_file.txt')
        print('\nScraping urls...')
        self.scraper.scrape_urls('medinify/scrapers/temp_urls_file.txt')
        self.data = self.data.append(self.scraper.dataset, ignore_index=True)

        os.remove('medinify/scrapers/temp_urls_file.txt')
        print('Collected reviews.')

    def collect_from_urls(self, urls_file):
        """
        Given a file listing drug urls, collects review data into Dataset
        :param urls_file: path to file listing drug urls
        """
        assert self.scraper, "In order to collect reviews, a scraper must be specified"

        print('\nScraping urls...')
        self.scraper.scrape_urls(urls_file)
        self.data = self.data.append(self.scraper.dataset, ignore_index=True)



