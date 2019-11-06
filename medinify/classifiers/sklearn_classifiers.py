"""
Creates subclasses of all sklearn classifiers used in Classifier, in order
to override methods to change the signatures
(The CNN learner required fit() and predict() function to have different
signatures, and it makes the code in Classifier cleaner and more generalized
to have only only call to model.learner.fit and model.learner.predict)

Also provides default processors so users aren't required to specify representation
"""
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


class NaiveBayesLearner(MultinomialNB):
    default_representation = 'bow'

    def fit(self, features, labels, model):
        super().fit(features, labels)

    def predict(self, features, model):
        super().predict(features)


class RandomForestLearner(RandomForestClassifier):
    default_representation = 'bow'

    def fit(self, features, labels, model):
        super().fit(features, labels)

    def predict(self, features, model):
        super().predict(features)


class SVCLearner(SVC):
    default_representation = 'embedding'

    def fit(self, features, labels, model):
        super().fit(features, labels)

    def predict(self, features, model):
        super().predict(features)


