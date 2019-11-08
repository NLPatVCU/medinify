"""
Examples for how to use the Medinify classifier functionality
"""

from medinify.classifiers import Classifier
from medinify.datasets import SentimentDataset


def main():
    clf = Classifier('nb')  # 'svm', 'cnn', or 'rf'
    dataset = SentimentDataset('<some csv>')
    model = clf.fit(dataset)
    clf.evaluate(model, trained_model_file='<some trained model')  # or trained_model = model
    clf.validate(dataset)
    clf.classify(dataset, output_file='<output file>', trained_model_file='<model file>')  # or trained_model = model


if __name__ == '__main__':
    main()

