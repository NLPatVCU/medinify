
from medinify.process import Processor
from sklearn.feature_extraction.text import CountVectorizer


class BowProcessor(Processor):

    nickname = 'bow'

    def __init__(self):
        super().__init__()
        self.vectorizer = CountVectorizer(tokenizer=self.tokenize)

    def get_features(self, dataset):
        try:
            self.vectorizer.vocabulary_
        except AttributeError:
            self.vectorizer.fit(dataset.data_table[dataset.text_column])
        count_vectors = self.vectorizer.transform(dataset.data_table[dataset.text_column])
        return count_vectors

