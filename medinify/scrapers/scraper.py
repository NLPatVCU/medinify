
from abc import ABC, abstractmethod
import pandas as pd
import os


class Scraper(ABC):

    data_collected = []
    dataset = None

    def __init__(self, collect_ratings=True, collect_dates=True, collect_drugs=True,
                 collect_user_ids=False, collect_urls=False):
        """
        The Scraper parent class defines the functions which any drug review scraper must implements
        :param collect_ratings: whether or not to collect ratings data
        :param collect_dates: whether or not to collect date posted data
        :param collect_drugs: whether or not to collect drug name data
        :param collect_user_ids: whether or not to collect user id data
        :param collect_urls: whether or not to collect drug url data
        """
        self.data_collected = []
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

    def get_urls(self, drug_names_file, output_file):
        """
        Given a text file of drug names, searches for and writes file with review urls
        :param drug_names_file: path to text file containing review urls
        :param output_file: path to file to output urls
        """
        review_urls = []
        unfound_drugs = []
        with open(drug_names_file, 'r') as f:
            for line in f.readlines():
                drug_name = line.strip()
                drug_review_urls = self.get_url(drug_name)
                if len(drug_review_urls) == 0:
                    unfound_drugs.append(drug_name)
                else:
                    review_urls.extend(drug_review_urls)
        with open(output_file, 'w') as url_f:
            for url in review_urls:
                url_f.write(url + '\n')
        print('Wrote review url file.')
        print('No urls found for {} drugs: {}'.format(len(unfound_drugs), unfound_drugs))

    def scrape_urls(self, urls_file, output_file, start=0):
        """
        Given a file containing a list of drug urls, scrapes those urls
        :param urls_file: path to text file containing drug urls
        :param output_file: file to output review data
        :param start: which url to start from
        """
        with open(urls_file, 'r') as f:
            urls = []
            for url in f.readlines():
                urls.append(url.strip())
        if os.path.exists(output_file):
            df = pd.read_csv(output_file)
            df.columns = self.data_collected
            self.dataset = df
        num_url = start
        for url in urls[start:]:
            print('Scraping url {} of {}'.format(num_url + 1, len(urls)))
            self.scrape(url)
            self.dataset.to_csv(output_file, index=False)
            print('\nSafe to quit. Start from {}.'.format(num_url + 1))
            num_url += 1




