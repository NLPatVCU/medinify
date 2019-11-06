
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


class NaiveBayesLearner(MultinomialNB):
    default_representation = 'bow'

    def fit(self, X, y, model):
        super().fit(X, y)

    def predict(self, X, model):
        super().predict(X)


class RandomForestLearner(RandomForestClassifier):
    default_representation = 'bow'

    def fit(self, X, y, model):
        super().fit(X, y)

    def predict(self, X, model):
        super().predict(X)


class SVCLearner(SVC):
    default_representation = 'embedding'

    def fit(self, X, y, model):
        super().fit(X, y)

    def predict(self, X, model):
        super().predict(X)


