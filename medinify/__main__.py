
"""
Medinify Command Line Interface Setup
"""

import argparse
from medinify.sentiment import Classifier
from medinify.datasets import Dataset


def setup_classifier(args):
    """
    Sets up classifier.
    :param args: Argparse args object.
    :return clf: classifier
    """
    clf = Classifier(args.classifier, data_representation=args.data_representation,
                     neg_threshold=args.neg_threshold, num_classes=args.num_classes,
                     pos_threshold=args.pos_threshold, pos=args.pos, w2v_file=args.word_embeddings)

    return clf


def setup_dataset(args):
    """
    Sets up dataset.
    :param args: command line arguments
    :return: Dataset
    """
    dataset = Dataset(scraper=args.scraper,
                      use_user_ids=args.collect_user,
                      use_urls=args.collect_url)
    return dataset


def train(args, clf):
    """
    Trains a model and saves it to a file
    :param args: command line arguments
    :param clf: classifier
    """
    clf.fit(args.output, reviews_file=args.reviews)


def evaluate(args, clf):
    """
    Evaluates a trained model
    :param args: command line arguments
    :param clf: classifier
    """
    clf.evaluate(args.model, eval_reviews_csv=args.reviews)


def validate(args, clf):
    """
    Runs cross validation
    :param args: command line arguments
    :param clf: classifier
    """
    clf.validate(args.reviews, args.folds)


def classify(args, clf):
    """
    Writes sentiment classifications file
    :param args: command line arguments
    :param clf: classifier
    """
    clf.classify(trained_model_file=args.model,
                 reviews_csv=args.reviews,
                 output_file=args.output)


def collect(args, dataset):
    """
    Scrapes reviews data
    :param args: command line arguments
    :param dataset: medinify dataset object
    """
    if args.names_file:
        dataset.collect_from_drug_names(args.names_file)
        dataset.write_file(args.output,
                           write_user_ids=args.collect_user,
                           write_urls=args.collect_url)
    elif args.urls_file:
        dataset.collect_from_urls(args.urls_file)
        dataset.write_file(args.output,
                           write_user_ids=args.collect_user,
                           write_urls=args.collect_url)


def main():
    """
    Main function where initial argument parsing happens.
    """
    # Argparse setup
    parser = argparse.ArgumentParser(prog='medinify', description='Drug Review Sentiment Analysis Tools.')
    parser.add_argument('-c', '--classifier', help='Classifier type.', default='nb', choices=['nb', 'rf', 'svm'])
    parser.add_argument('-dr', '--data-representation', default='count',
                        help='How processed comment data will be numerically represented.',
                        choices=['count', 'tfidf', 'pos', 'embedding'])
    parser.add_argument('-n', '--num-classes', default=2, help='Number of rating classes.', choices=[2, 3, 5], type=int)
    parser.add_argument('-pt', '--pos-threshold', default=4.0, help='Positive rating threshold.')
    parser.add_argument('-nt', '--neg-threshold', default=2.0, help='Negative rating threshold.')
    parser.add_argument('-wv', '--word-embeddings', help='Path to file containing pretrained word2vec embeddings (If using average emeddings as data representation).')
    parser.add_argument('-p', '--pos', help='Part of speech (If using part of speech count vectors as data representation)')
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
    parser_collect.set_defaults(func=collect)

    # Parse initial args
    args = parser.parse_args()

    # Run proper function
    if args.func in [train, evaluate, validate, classify]:
        clf = setup_classifier(args)
        args.func(args, clf)
    elif args.func == collect:
        dataset = setup_dataset(args)
        args.func(args, dataset)


if __name__ == '__main__':
    main()


