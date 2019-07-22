
from abc import ABC, abstractmethod
import pandas as pd


class Scraper(ABC):

    data_collected = []
    dataset = None

    def __init__(self, collect_ratings=True, collect_dates=True, collect_drugs=True,
                 collect_user_ids=False, collect_urls=False):

        self.data_collected.append('comment')
        if collect_ratings:
            self.data_collected.append('rating')
        if collect_dates:
            self.data_collected.append('date')
        if collect_drugs:
            self.data_collected.append('drug')
        if collect_user_ids:
            self.data_collected.append('user id')
        if collect_urls:
            self.data_collected.append('url')

        self.dataset = pd.DataFrame(columns=self.data_collected)

    @abstractmethod
    def scrape_page(self, url):
        """
        Scrapes a single page of drug reviews
        :param url: drug reviews page url
        :return:
        """
        pass

    @abstractmethod
    def scrape(self, url):
        """
        Scrapes all reviews of a given drug
        :param url: drug reviews url
        """
        pass

    @abstractmethod
    def get_url(self, drug_name):
        """
        Given a drug name, finds the drug review page(s) on a given review forum
        :param drug_name: name of drug being searched for
        :return: drug url on given review forum
        """
        pass

    @abstractmethod
    def get_urls(self, drug_urls_file, output_file):
        """
        Given a text file of drug names, searches for and writes file with review urls
        :param drug_urls_file: path to text file containing review urls
        :param output_file: path to file to output urls
        """
        pass
