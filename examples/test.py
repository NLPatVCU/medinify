from medinify.sentiment.review_classifier import ReviewClassifier

if __name__ == "__main__":
    review_classifier = ReviewClassifier('nb', 'stopwords.txt', 10)
    review_classifier.train('citalopram_train.csv')
    review_classifier.classify('neutral.txt')