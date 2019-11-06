"""
Medinify scrapers facilitate the collection of drug review data from online forums
Currently, Medinify can specifically scrape certain data from the following websites:

    --> WebMD.com
        -> Comments (Review text)
        -> 5-Point Scale Star Ratings ('Effectiveness', 'Ease of Use', and 'Satisfaction')
        -> Post Dates
        -> Use IDs
    --> Drugs.com
        -> Comments (Review text)
        -> Star Rating (0.0-10.0 Point Scale)
        -> Post Dates
        -> Use IDs
    --> DrugRatingz.com
        -> Comments (Review text)
        -> 5-Point Scale Star Ratings ('Effectiveness', 'Convenience', 'No Side Effects', and 'Value')
        -> Post Dates
    --> EverydayHealth.com
        -> Comments (Review text)
        -> 5-Point Scale Star Rating
        -> Post Dates

    (Additionally, review URLs and associated drug names can be stored alongside each review's scraped data)
"""
from abc import ABC, abstractmethod


class Scraper(ABC):
    """
    The Scraper abstract class describes the required functionality of any drug forum scraper
    and implements some functionality that is identical across all scrapers

    Attributes:
        collect_urls:   (Boolean) Whether or not to collect each review's associated url
        reviews:        (list[dict]) Scraped review data
    """
    def __init__(self, collect_urls=False):
        """
        Standard constructor for all drug forum scrapers
        :param collect_urls: (Boolean) whether or not to collects the urls associated with each review
        """
        self.collect_urls = collect_urls
        self.reviews = []

    @abstractmethod
    def scrape_page(self, url):
        """
        Function for collecting the reviews data from one page of a drug review forum
        Scraped data is stored in scraper's review attribute
        :param url: (str) url for the particular website being scraped
        """
        pass

    @abstractmethod
    def scrape(self, url):
        """
        Scrapes all the review data for a particular drug on a particular drug review forum
        Scraped data is stored in scraper's review attribute
        (If scraper already has scraped review data, it is discarded before continued scraping)
        :param url: (str) url for the first page of reviews for this drug
        """
        if len(self.reviews) > 0:
            print('Clearing scraper\'s pre-existent dataset of {} '
                  'collected reviews...'.format(len(self.reviews)))
            self.reviews = []

    @abstractmethod
    def get_url(self, drug_name):
        """
        Searches drug forum for reviews for a certain drug
        :param drug_name: (str) drug name
        :return: drug url (str) if found, None if not found
        """
        pass

    def get_urls(self, drug_names_file, output_file=None):
        """
        Given a text file containing drug names (one name per line), collects
        the urls for each drug name, and either writes those urls to a file,
        one url per line (if an output file is specified), or returns urls as a list[str]
        :param drug_names_file: (str) path to file containing drug names
        :param output_file: (str) path to output drug urls file
        """
        review_urls = []
        not_found_drugs = []
        with open(drug_names_file, 'r') as f:
            for line in f.readlines():
                drug_name = line.strip()
                drug_review_url = self.get_url(drug_name)
                if drug_review_url:
                    review_urls.append(drug_review_url)
                else:
                    not_found_drugs.append(drug_name)
        if output_file:
            with open(output_file, 'w') as url_f:
                for url in review_urls:
                    url_f.write(url + '\n')
            print('Wrote review url file.')
            print('No urls found for %d drugs: %s' % (
                len(not_found_drugs), ', '.join(not_found_drugs)))
        else:
            print('No urls found for %d drugs: %s' % (
                len(not_found_drugs), ', '.join(not_found_drugs)))
            return review_urls





