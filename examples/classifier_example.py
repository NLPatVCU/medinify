"""
Examples for how to use the Medinify package
"""

from medinify.sentiment.review_classifier import ReviewClassifier

def main():
    """ Main function.
    """

    # Naive bayes classifier
    review_classifier = ReviewClassifier('nb', 'stopwords.txt')
    review_classifier.train('citalopram_train.csv')
    review_classifier.classify('neutral.txt')

    # Decision tree classifier
    # review_classifier = ReviewClassifier('dt', 'stopwords.txt')
    # review_classifier.train('citalopram_train.csv')
    # review_classifier.classify('neutral.txt')

if __name__ == "__main__":
    main()
