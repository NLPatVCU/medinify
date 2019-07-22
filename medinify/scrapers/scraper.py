
from abc import ABC, abstractmethod
import pandas as pd


class Scraper(ABC):

    data_collected = []
    dataset = None

    def __init__(self, collect_ratings=True, collect_dates=True, collect_drugs=True,
                 collect_user_ids=False, collected_urls=False):

        self.data_collected.append('comment')
        if collect_ratings:
            self.data_collected.append('rating')
        if collect_dates:
            self.data_collected.append('date')
        if collect_drugs:
            self.data_collected.append('drug')
        if collect_user_ids:
            self.data_collected.append('user id')
        if collected_urls:
            self.data_collected.append('url')

        self.dataset = pd.DataFrame(columns=self.data_collected)