
import pandas as pd
from medinify.scrapers.webmd_scraper import WebMDScraper
from medinify.scrapers.drugs_scraper import DrugsScraper
from medinify.scrapers.drugratingz_scraper import DrugRatingzScraper
from medinify.scrapers.everydayhealth_scraper import EverydayHealthScraper


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


