from medinify.datasets.process import Processor
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


class BowProcessor(Processor):

    def __init__(self):
        super().__init__()
        self.vectorizer = CountVectorizer(tokenizer=self.tokenize)

    def get_features(self, dataset):
        dataset = remove_neutral(dataset)
        if dataset.num_classes == 2:
            dataset.data_table = dataset.data_table.loc[
                dataset.data_table[dataset.feature_column] != 3.0]
        try:
            self.vectorizer.vocabulary_
        except AttributeError:
            self.vectorizer.fit(dataset.data_table[dataset.text_column])
        count_vectors = self.vectorizer.transform(dataset.data_table[dataset.text_column])
        return count_vectors

    def get_labels(self, dataset):
        dataset = remove_neutral(dataset)
        dataset.data_table['label'] = dataset.data_table[dataset.feature_column].apply(
            lambda x: self._rating_to_label(x, dataset.num_classes))
        return dataset.data_table['label']

    @staticmethod
    def _rating_to_label(rating, num_classes):
        if num_classes == 2:
            if rating in [1.0, 2.0]:
                return 0
            elif rating in [4.0, 5.0]:
                return 1
            else:
                return np.NaN
        elif num_classes == 3:
            if rating in [1.0, 2.0]:
                return 0
            elif rating in [4.0, 5.0]:
                return 2
            else:
                return 1


def remove_neutral(dataset):
    if dataset.num_classes == 2:
        dataset.data_table = dataset.data_table.loc[
            dataset.data_table[dataset.feature_column] != 3.0]
    return dataset
