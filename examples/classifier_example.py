"""
Examples for how to use the Medinify package
"""

from medinify.sentiment import ReviewClassifier, CNNReviewClassifier
import sys

def main():
    """ Main function.
    """
    # Review sentiment classifier
    # review_classifier = ReviewClassifier('nb') # Try 'nb', 'dt', 'rf', and 'nn'
    # review_classifier.train('citalopram-reviews.csv')
    # review_classifier.save_model()
    # review_classifier.load_model()
    # review_classifier.evaluate_average_accuracy('citalopram-reviews.csv')
    # review_classifier.classify('neutral.txt')

    input_file = sys.argv[1]

    sent = CNNReviewClassifier()
    sent.evaluate_k_fold(input_file, 5)

if __name__ == "__main__":
    main()
