"""
Examples for how to use the Medinify package
"""

from medinify.sentiment import ReviewClassifier
# from medinify.sentiment import NeuralNetReviewClassifier

def main():
    """ Main function.
    """

    # Naive bayes classifier
    review_classifier = ReviewClassifier('nb', 'stopwords.txt')
    review_classifier.train('citalopram-reviews.csv')
    # review_classifier.classify('neutral.txt')

    # Decision tree classifier
    # review_classifier = ReviewClassifier('dt', 'stopwords.txt')
    # review_classifier.train('citalopram_train.csv')
    # review_classifier.classify('neutral.txt')

    # Neural Network classifier
    # review_classifier = NeuralNetReviewClassifier()
    # review_classifier.train('citalopram_train.csv')
    # review_classifier.classify('neutral.txt')


if __name__ == "__main__":
    main()
