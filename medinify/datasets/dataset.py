
import pandas as pd


class Dataset:

    data_used = []
    data = None

    def __init__(self, use_rating=True, use_dates=True,
                 use_drugs=True, use_user_ids=False,
                 use_urls=False):
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

        self.data = pd.DataFrame(columns=self.data_used)
