"""
Text Classifier primarily for medical text. Currently can use Naive Bayes, Neural Network, or Decision Tree for sentiment analysis.

Based on work by Amy Olex 11/13/17.
"""

import string
import numpy
from nltk.classify import NaiveBayesClassifier
from nltk.classify import DecisionTreeClassifier
import nltk.classify.util
import nltk
import math
import csv
import argparse
import random
from collections import Counter
from sklearn.model_selection import StratifiedKFold


def format_text(text, stop_words=None):
    """ Takes a string, converts to lowercase, and removes all punctuation & stop words.

    Args:
      sent: String to be converted.
      stop_words: Stop words to be removed. Defaults to None.
    Returns:
      The formatted text.
    """


    text = clean_text(text)

    if stop_words is not None:
        text = remove_stop_words(text, stop_words)
    
    return ({word: True for word in nltk.word_tokenize(text)})

def remove_stop_words(text, stop_words):
    """ Removes all stop words from text.

    Args:
        text: Text to be stripped of stop words.
        stop_words: Stop words to strip from text.
    Returns:
        Stripped text.
    """

    word_list = text.split()
    filtered_words = [word for word in word_list if word not in stop_words]
    text = ' '.join(filtered_words)

    return text

def clean_text(text):
    """ Takes a string, converts to lowercase, and removes all punctuation.

    Args:
      sent: String to be converted.
      stop_words: Stop words to be removed. Defaults to None.
    Returns:
      The cleaned text.
    """

    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator).lower()

    return text

def parse_comments(comment_file):
    """ Parses a CSV of comments into a list of comments with rating.

    Args:
        comment_file: Comment file to be parsed.
    Returns:
        The list of comments.
    """

    comments = []

    with open(comment_file, newline='') as cf:
        reader = csv.DictReader(cf)

        for row in reader:
            comments.append({'comment': row['comment'], 'rating': row['rating']})

    return comments


if __name__ == "__main__":
    
    ## Parse input arguments
    parser = argparse.ArgumentParser(description='Train a Sentiment Classifier')
    parser.add_argument('-i', metavar='inputfile', type=str, help='path to the input csv file for training and testing.', required=True)
    parser.add_argument('-c', metavar='toclassify', type=str, help='path to file with entries to classify.', required=False, default=None)
    parser.add_argument('-s', metavar='stopwords', type=str, help='path to stopwords file', required=True)
    parser.add_argument('-p', metavar='posratings', type=str, help='a list of positive ratings as strings', required=False, default=4.0)
    parser.add_argument('-n', metavar='negratings', type=str, help='a list of negative ratings as strings', required=False, default=2.0)
    parser.add_argument('-z', metavar='iterations', type=str, help='the number of times to repeat the classifier training', required=False, default=1)   
    parser.add_argument('-d', metavar='domain', type=str, help='a file with text from a different domain.', required=False, default = None)
    parser.add_argument('-cl', metavar='classifier', type=str, help='classifer to use', required=False, default='nb')
    args = parser.parse_args()
    
    ## Parse data from files
    comments = parse_comments(args.i)

    with open('stopwords.txt') as f:
        text = clean_text(f.read())
        stop_words = text.splitlines()
    
    ## Parse and convert positive and negative examples.
    pos_list=[]
    neg_list=[]

    for comment in comments:
        body = comment['comment']
        rating = comment['rating']

        body = format_text(body, stop_words)

        if float(rating) <= float(args.n):
            neg_list.append((body, 'neg'))
        if float(rating) >= float(args.p):
            pos_list.append((body, 'pos'))

    seed = 123
    numpy.random.seed(seed)
    print("Total Negative Instances:" + str(len(neg_list)) + "\nTotal Positive Instances:" + str(len(pos_list)))

    negcutoff = math.floor(len(neg_list) * 1)
    poscutoff = math.floor(len(pos_list) * 1)
    neg_idx_train = sorted(random.sample(range(len(neg_list)), negcutoff))
    neg_train = [neg_list[i] for i in neg_idx_train]

    pos_idx_train = sorted(random.sample(range(len(pos_list)), poscutoff))
    pos_train = [pos_list[i] for i in pos_idx_train]

    dataset = neg_train + pos_train

    X = [x[0] for x in dataset]
    Y = [x[1] for x in dataset]
    kfold = StratifiedKFold(n_splits=int(args.z), shuffle=True, random_state=seed)
    cvscores = []
    for train, test in kfold.split(X,Y):
        # print(dataset[train[0]])
        train_data = []
        for i in range(len(train)):
            train_data.append(dataset[train[i]])
        test_data = []
        for i in range(len(test)):
            test_data.append(dataset[test[i]])

        if (args.cl == 'nb'):
            model = NaiveBayesClassifier.train(train_data)
        elif (args.cl == 'dt'):
            model = DecisionTreeClassifier.train(train_data)

        scores = nltk.classify.util.accuracy(model, test_data)
        print("{}%".format(scores * 100))
        cvscores.append(scores * 100)
        # plot_model(model, to_file='model.png')

        if (args.cl == 'nb'):
            model.show_most_informative_features()

    print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))
    
    ### Import the file needing classification.
    if args.c is not None:
        with open(args.c) as file:
            toclass = file.readlines()

        for sent in toclass:
            print(str(model.classify(format_text(sent))) + " :: " + sent)

    if args.d is not None:
        print("ARGS D")
        domain_list = []
        with open(args.d) as domainfile:
            reader = csv.DictReader(domainfile)
            for row in reader:
                domain_list.append({'comment': row['comment'], 'rating': row['rating']})
        print(str(len(domain_list)))

        d_list = []
        for c in range(len(domain_list)):
            tmp_c = domain_list[c]['comment']
            tmp_r = domain_list[c]['rating']

            if tmp_r in args.n:
                d_list.append((format_text(tmp_c, stop_words), 'neg'))
            if tmp_r in args.p:
                d_list.append((format_text(tmp_c, stop_words), 'pos'))

        # classifier2 = NaiveBayesClassifier.train(domain_list)
        model = NaiveBayesClassifier.train(dataset)
        domain_accuracy = nltk.classify.util.accuracy(model, d_list)
        print('Classifier domain shift accuracy:', domain_accuracy)