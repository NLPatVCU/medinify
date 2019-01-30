"""Configure medinify.scrapers
"""
from medinify.sentiment.review_classifier import ReviewClassifier
from medinify.sentiment.review_classifier import NeuralNetReviewClassifier

__all__ = (
    'ReviewClassifier',
    'NeuralNetReviewClassifier',
)
