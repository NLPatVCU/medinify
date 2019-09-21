
from abc import ABC, abstractmethod


class Scraper(ABC):

    def __init__(self, collect_user_ids=False, collect_urls=False):
        """
        The Scraper parent class defines the functions which any drug review scraper must implements
        :param collect_user_ids: whether or not to collect user id data
        :param collect_urls: whether or not to collect drug url data
        """
        self.data_collected = ['comment', 'rating', 'date', 'drug']
        if collect_user_ids:
            self.data_collected.append('user id')
        if collect_urls:
            self.data_collected.append('url')

        self.reviews = []

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
        if len(self.reviews) > 0:
            print('Clearing scraper\'s pre-existent dataset of {} '
                  'collected reviews...'.format(len(self.reviews)))
            self.reviews = []

    @abstractmethod
    def get_url(self, drug_name, return_multiple=False):
        """
        Given a drug name, finds the drug review page(s) on a given review forum
        :param drug_name: name of drug being searched for
        :param return_multiple: if multiple urls are found, whether or not to return all of them
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
                drug_review_urls = self.get_url(drug_name, return_multiple=True)
                if drug_review_urls:
                    review_urls.extend(drug_review_urls)
                else:
                    unfound_drugs.append(drug_name)
        with open(output_file, 'w') as url_f:
            for url in review_urls:
                url_f.write(url + '\n')
        print('Wrote review url file.')
        print('No urls found for %d drugs: %s' % (len(unfound_drugs), ', '.join(unfound_drugs)))





