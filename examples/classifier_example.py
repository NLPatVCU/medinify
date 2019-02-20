"""
Examples for how to use the Medinify package
"""

from medinify.sentiment import ReviewClassifier

def main():
    """ Main function.
    """

    # Review sentimet classifier
    review_classifier = ReviewClassifier('nb') # Try 'nb', 'dt', and 'nn'
    review_classifier.evaluate_average_accuracy('citalopram-reviews.csv')
    # review_classifier.train('citalopram-reviews.csv')

if __name__ == "__main__":
    main()
