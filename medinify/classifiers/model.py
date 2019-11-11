
import pickle
from medinify.vectorizers.utils import get_lookup_table
from medinify.vectorizers.utils import find_embeddings
from gensim.models import KeyedVectors
from medinify.classifiers import CNNLearner, CNNClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from medinify import vectorizers

MultinomialNB.default_vectorizer = 'bow'
SVC.default_vectorizer = 'embedding'
RandomForestClassifier.default_vectorizer = 'bow'
CNNLearner.default_vectorizer = 'matrix'


class Model:
    """
    Model class contains a learner (some classifier, like Naive Bayes, Random Forest,
    etc.) and a vectorizer for transforming text data into numeric data

    These have to be put together into the same object because all data used to
    for fitting, evaluating, and classifying with the learner has to be vectorized in
    the same way
    """
    def __init__(self, learner='nb', representation=None):
        """
        Constructor for Model
        :param learner: (str) classifier type ('nb' - Naive Bayes, 'rf' - Random Forest,
            'svm' - Support Vector Machine, 'cnn' - Convolutional Neural Network)
        :param representation: How text data will be vectorized ('bow' -
            bag of words, 'embedding' - average embedding, 'matrix' - embedding matrix)
        """
        self.type = learner
        if learner == 'nb':
            self.learner = MultinomialNB()
        elif learner == 'rf':
            self.learner = RandomForestClassifier(
                n_estimators=100, criterion='gini', max_depth=None, bootstrap=False, max_features='auto')
        elif learner == 'svm':
            self.learner = SVC(kernel='rbf', C=10, gamma=0.01)
        elif learner == 'cnn':
            self.learner = CNNLearner()
        else:
            raise AssertionError('model_type must by \'nb\', \'svm\', \'rf\', or \'cnn\'')

        for vec in vectorizers.Vectorizer.__subclasses__():
            if representation and vec.nickname == representation:
                self.vectorizer = vec()
            elif vec.nickname == self.learner.default_vectorizer:
                self.vectorizer = vec()
        try:
            self.vectorizer
        except AttributeError:
            print('Invalid feature representation')

    def save_model(self, path):
        """
        Saves trained model to a file
        :param path: (str) file name to save at (all models will be saved to models/ directory)
        """
        with open(path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
            if self.type == 'cnn':
                pickle.dump(self.learner.network.state_dict(), f)
            else:
                pickle.dump(self.learner, f)

    def load_model(self, path):
        """
        Load trained model from file
        :param path: (str) model file name (will search through models/ directory for this file)
        """
        with open(path, 'rb') as f:
            self.vectorizer = pickle.load(f)
            if self.type == 'cnn':
                state_dict = pickle.load(f)
                embeddings_file = find_embeddings()
                w2v = KeyedVectors.load_word2vec_format(embeddings_file)
                lookup_table = get_lookup_table(w2v)
                network = CNNClassifier(lookup_table)
                network.load_state_dict(state_dict)
                self.learner.network = network
            else:
                self.learner = pickle.load(f)
