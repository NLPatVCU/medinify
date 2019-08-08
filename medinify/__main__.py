
"""
Medinify Command Line Interface Setup
"""

import argparse
from medinify.sentiment import Classifier
from medinify.datasets import Dataset
from medinify import config


def configure(args):
    """
    Configures constant variables
    :param args: command line arguments
    """
    config.POS_THRESHOLD = args.pos_threshold
    config.NEG_THRESHOLD = args.neg_threshold
    config.NUM_CLASSES = args.num_classes
    config.DATA_REPRESENTATION = args.data_representation
    config.RATING_TYPE = args.rating_type


def train(args):
    """
    Trains a model and saves it to a file
    :param args: command line arguments
    """
    clf = Classifier(classifier_type=args.classifier, w2v_file=args.word_embeddings, pos=args.pos)
    clf.fit(output_file=args.output, reviews_file=args.reviews)


def evaluate(args):
    """
    Evaluates a trained model
    :param args: command line arguments
    """
    clf = Classifier(classifier_type=args.classifier, w2v_file=args.word_embeddings, pos=args.pos)
    clf.evaluate(args.model, eval_reviews_csv=args.reviews)


def validate(args):
    """
    Runs cross validation
    :param args: command line arguments
    """
    clf = Classifier(classifier_type=args.classifier, w2v_file=args.word_embeddings, pos=args.pos)
    clf.validate(args.reviews, temp_file_name=args.temp_file, k_folds=args.folds)


def classify(args):
    """
    Writes sentiment classifications file
    :param args: command line arguments
    """
    clf = Classifier(classifier_type=args.classifier, w2v_file=args.word_embeddings, pos=args.pos)
    clf.classify(trained_model_file=args.model,
                 reviews_csv=args.reviews,
                 output_file=args.output)


def collect(args):
    """
    Scrapes reviews data
    :param args: command line arguments
    """
    dataset = Dataset(args.scraper, use_user_ids=args.collect_user, use_urls=args.collect_url)

    if args.names_file:
        dataset.collect_from_drug_names(args.names_file, args.output, start=args.start)
    elif args.urls_file:
        dataset.collect_from_urls(args.urls_file, args.output, start=args.start)


def main():
    """
    Main function where initial argument parsing happens.
    """
    # Argparse setup
    parser = argparse.ArgumentParser(prog='medinify', description='Drug Review Sentiment Analysis Tools.')
    parser.add_argument('-pt', '--pos-threshold', help='Ratings Positive Threshold.', default=4.0, type=float)
    parser.add_argument('-nt', '--neg-threshold', help='Ratings Negative Threshold.', default=2.0, type=float)
    parser.add_argument('-nc', '--num-classes', help='Number of ratings classes', default=2,
                        type=int, choices=[2, 3, 5])
    parser.add_argument('-d', '--data-representation', help='How comment data should be represented numerically',
                        default='count', choices=['count', 'tfidf', 'embeddings', 'pos'])
    parser.add_argument('-t', '--rating-type', help='If dataset contains multiple types of ratings, which one to use',
                        default='effectiveness')
    parser.add_argument('-c', '--classifier', help='Classifier type', default='nb', choices=['nb', 'rf', 'svm'])
    parser.add_argument('-wv', '--word-embeddings',
                              help='Path to word embeddings file if using average embeddings', default=None)
    parser.add_argument('-p', '--pos', help='Part of speech if using part of speech count vectors', default=None)
    subparsers = parser.add_subparsers()

    # Train arguments
    parser_train = subparsers.add_parser('train', help='Train a new model.')
    parser_train.add_argument('-r', '--reviews', help='Path to reviews file to train on.', required=True)
    parser_train.add_argument('-o', '--output', help='Path to save model file', required=True)
    parser_train.set_defaults(func=train)

    # Evaluate arguments
    parser_eval = subparsers.add_parser('evaluate', help='Evaluate a trained model.')
    parser_eval.add_argument('-r', '--reviews', help='Path to reviews file to train on.', required=True)
    parser_eval.add_argument('-m', '--model', help='Path to saved model file', required=True)
    parser_eval.set_defaults(func=evaluate)

    # Validate arguments
    parser_valid = subparsers.add_parser('validate', help='Cross validate a model.')
    parser_valid.add_argument('-r', '--reviews', help='Path to reviews file to train on.', required=True)
    parser_valid.add_argument('-f', '--folds', help='Number of folds.', required=True, type=int)
    parser_valid.add_argument('-tf', '--temp-file', help='Where to save temporary model files.', required=True)
    parser_valid.set_defaults(func=validate)

    # Classify arguments
    parser_classify = subparsers.add_parser('classify', help='Classifies the sentiment of reviews.')
    parser_classify.add_argument('-r', '--reviews', help='Path to reviews file to train on.', required=True)
    parser_classify.add_argument('-m', '--model', help='Path to saved model file', required=True)
    parser_classify.add_argument('-o', '--output', help='Path to save model file', required=True)
    parser_classify.set_defaults(func=classify)

    # Collect arguments
    parser_collect = subparsers.add_parser('collect', help='Collects drug review data.')
    parser_collect.add_argument('-s', '--scraper', help='If collecting data, which scraper to use',
                                choices=['WebMD', 'Drugs', 'DrugRatingz', 'EverydayHealth'])
    file_path = parser_collect.add_mutually_exclusive_group(required=True)
    file_path.add_argument('-nf', '--names-file',
                           help='Path to drug names file.')
    file_path.add_argument('-uf', '--urls-file',
                           help='Path to urls file.')
    parser_collect.add_argument('--collect-user', help='Whether to collect user id data', action='store_true')
    parser_collect.add_argument('--collect-url', help='Whether to collect url data', action='store_true')
    parser_collect.add_argument('-o', '--output', help='Path to save model file', required=True)
    parser_collect.add_argument('-st', '--start', help='Where to start collecting if continuing collection',
                                default=0, type=int)
    parser_collect.set_defaults(func=collect)

    # Parse initial args
    args = parser.parse_args()
    configure(args)

    # Run proper function
    args.func(args)


if __name__ == '__main__':
    main()


