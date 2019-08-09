"""
Examples for how to use the Medinify package
"""

from medinify.sentiment import Classifier
from medinify.sentiment.cnn_review_classifier import CNNReviewClassifier
import sys


def main():
    """
    Examples of how to use the medinify Classifier class
    """

    reviews_file = sys.argv[1]
    rating_type = sys.argv[2]

    clf = CNNReviewClassifier('new_w2v.model')
    clf.evaluate_k_fold(input_file=reviews_file, num_folds=10, num_epochs=20, rating_type=rating_type)

    """
    clf = Classifier('nb')  # 'nb', 'svm', or 'rf'
    clf.fit(output_file='examples/nb_model.pkl', reviews_file='examples/reviews_file.csv')
    clf.evaluate('examples/trained_model.pkl', 'exaples/evaluation_reviews.csv')
    clf.classify('examples/trained_model.pkl', output_file='examples/classified.txt',
                 reviews_csv='examples/reviews.csv')
    clf.validate(review_csv='examples/reviews.csv', k_folds=5)
    """


if __name__ == "__main__":
    main()
