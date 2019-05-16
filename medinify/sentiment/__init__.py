"""Configure medinify.scrapers
"""
from medinify.sentiment.review_classifier import ReviewClassifier
from medinify.sentiment.cnn_review_classifier import CNNReviewClassifier, SentimentNetwork

__all__ = (
    'ReviewClassifier',
    'CNNReviewClassifier',
    'SentimentNetwork'
)
