
"""
Medinify Command Line Interface Setup
"""

import argparse
from medinify.sentiment import Classifier


def setup(args):
    """
    Sets up dataset and pipeline/model since it gets used by every command.
    :param args: Argparse args object.
    :return dataset, model: The dataset and model objects created.
    """
    clf = Classifier(args.classifier, data_representation=args.data_representation,
                     neg_threshold=args.neg_threshold, num_classes=args.num_classes,
                     pos_threshold=args.pos_threshold, pos=args.pos, w2v_file=args.word_embeddings)

    return clf


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
    parser.add_argument('-wv', '--word-embeddings', help='Path to file containing pretrained word2vec embeddings ''(If using average emeddings as data representation).')
    parser.add_argument('-p', '--pos', help='Part of speech ''(If using part of speech count vectors as data representation)')
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

    # Parse initial args
    args = parser.parse_args()

    # Run proper function
    clf = setup(args)
    args.func(args, clf)


if __name__ == '__main__':
    main()


