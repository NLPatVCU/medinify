"""
Examples for how to use the Medinify package
"""

from medinify.sentiment.review_classifier import ReviewClassifier
from medinify.sentiment.nn_review_classifier import NeuralNetReviewClassifier

def main():
    """ Main function.
    """

    #Naive bayes classifier
    review_classifier = ReviewClassifier('nb', 'stopwords.txt')
    review_classifier.train('citalopram_train.csv')
    review_classifier.classify('neutral.txt')

    # Decision tree classifier
    # review_classifier = ReviewClassifier('dt', 'stopwords.txt')
    # review_classifier.train('citalopram_train.csv')
    # review_classifier.classify('neutral.txt')

    # Neural Network classifier
    # review_classifier = NeuralNetReviewClassifier('stopwords.txt')
    # review_classifier.nn_train('citalopram_train.csv')

    # Still working on the classification piece
    # review_classifier.nn_classify('neutral.txt')


if __name__ == "__main__":
    main()
