### NBSentiment
### Author: Amy Olex
### 11/13/17
### This program takes in two csv files containing text to be classified as positive or negative sentiment.
### The first file contains data for training and testing, the second contains data to be classified.
### 

import string

import numpy
from nltk.classify import NaiveBayesClassifier
import nltk.classify.util
import nltk
import math
import csv
import argparse
import random
from collections import Counter
from sklearn.model_selection import StratifiedKFold


## Formats a string for input in the NB Classifier by converting all to lowercase and removing al punctuation.
# @author Amy Olex
# @param sent The string to be formatted.
# @param stopwords A list of stopwords to be removed. Default is None.
# @return A dictionary of each word as the key and True as the value.
def format_sentence(sent, stopwords=None):
    # convert to lowercase
    sent = sent.translate(str.maketrans("", "", string.punctuation)).lower()
    #remove stopwords
    if stopwords is not None:
        com_list = sent.split()
        filtered_words = []
        for word in com_list:
            if word not in stopwords:
                filtered_words.append(word)
        sent = ' '.join(filtered_words)
    
    return({word: True for word in nltk.word_tokenize(sent)})

#####
## End Function
#####




if __name__ == "__main__":
    
    ## Parse input arguments
    parser = argparse.ArgumentParser(description='Train a NB Sentiment Classifier')
    parser.add_argument('-i', metavar='inputfile', type=str, help='path to the input csv file for training and testing.', required=True)
    parser.add_argument('-c', metavar='toclassify', type=str, help='path to file with entries to classify.', required=False, default=None)
    parser.add_argument('-s', metavar='stopwords', type=str, help='path to stopwords file', required=True)
    parser.add_argument('-p', metavar='posratings', type=str, help='a list of positive ratings as strings', required=False, default=4.0)
    parser.add_argument('-n', metavar='negratings', type=str, help='a list of negative ratings as strings', required=False, default=2.0)
    parser.add_argument('-z', metavar='iterations', type=str, help='the number of times to repeat the classifier training', required=False, default=1)   
    parser.add_argument('-d', metavar='domain', type=str, help='a file with text from a different domain.', required=False, default = None)   
    
    args = parser.parse_args()
    
    
    
    ## Import csv file
    my_list = []
    with open(args.i) as commentfile:
        reader = csv.DictReader(commentfile)
        for row in reader:
            my_list.append({'comment': row['comment'], 'rating': row['rating']})
    
    ## Parse and convert positive and negative examples.
    pos_list=[]
    neg_list=[]
    for c in my_list:
        tmp_com = c['comment']
        tmp_rating = c['rating']

        #remove stop words
        with open(args.s) as raw:
            stopwords = raw.read().translate(str.maketrans("", "", string.punctuation)).splitlines()
 
            if float(tmp_rating) <= float(args.n):
                neg_list.append((format_sentence(tmp_com, stopwords), 'neg'))
            if float(tmp_rating) >= float(args.p):
                pos_list.append((format_sentence(tmp_com, stopwords), 'pos'))

    seed = 123
    numpy.random.seed(seed)
    print("Total Negative Instances:"+str(len(neg_list))+"\nTotal Positive Instances:"+str(len(pos_list)))

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
        model = NaiveBayesClassifier.train(train_data)
        scores = nltk.classify.util.accuracy(model, test_data)
        print("{}%".format(scores * 100))
        cvscores.append(scores * 100)
        # plot_model(model, to_file='model.png')
        model.show_most_informative_features()

    print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))

    ### create training and test sets
    ## set the cutoffs
    # negcutoff = math.floor(len(neg_list)*3/4)
    # poscutoff = math.floor(len(pos_list)*3/4)
    #
    # top10list = []
    # avgAccuracy = 0
    # for z in range(int(args.z)):
    #     #train = neg_list[:negcutoff] + pos_list[:poscutoff]
    #     #test = neg_list[negcutoff:] + pos_list[poscutoff:]
    #     neg_idx_train = sorted(random.sample(range(len(neg_list)), negcutoff))
    #     neg_train = [neg_list[i] for i in neg_idx_train]
    #
    #     neg_idx_test = set(range(len(neg_list))) - set(neg_idx_train)
    #     neg_test = [neg_list[i] for i in neg_idx_test]
    #
    #
    #     pos_idx_train = sorted(random.sample(range(len(pos_list)), poscutoff))
    #     pos_train = [pos_list[i] for i in pos_idx_train]
    #
    #     pos_idx_test = set(range(len(pos_list))) - set(pos_idx_train)
    #     pos_test = [pos_list[i] for i in pos_idx_test]
    #
    #     train = neg_train + pos_train
    #     test = neg_test + pos_test
    #     print('Training on %d instances, testing on %d instances' % (len(train), len(test)))
    #
    #     classifier = NaiveBayesClassifier.train(train)
    #     accuracy = nltk.classify.util.accuracy(classifier, test)
    #     avgAccuracy = avgAccuracy + accuracy
    #     print('Classifier accuracy:', accuracy)
    #     classifier.show_most_informative_features()
    #
    #     t10 = classifier.most_informative_features(10)
    #     tlist = [i[0] for i in t10]
    #     top10list = top10list + tlist
    
        ### Import the file needing classification.
    if args.c is not None:
        with open(args.c) as file:
            toclass = file.readlines()

        for sent in toclass:
            print(str(model.classify(format_sentence(sent))) + " :: " + sent)
    
    ### Count the occurences of each word that appeared in the top 10 over the 20 runs.
    # print("Average Accuracy: "+ str(avgAccuracy/int(args.z)))
    # my_counts = Counter(top10list)
    # print(my_counts)


    if args.d is not None:
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
                d_list.append((format_sentence(tmp_c, stopwords), 'neg'))
            if tmp_r in args.p:
                d_list.append((format_sentence(tmp_c, stopwords), 'pos'))

        # classifier2 = NaiveBayesClassifier.train(domain_list)
        model = NaiveBayesClassifier.train(dataset)
        domain_accuracy = nltk.classify.util.accuracy(model, d_list)
        print('Classifier domain shift accuracy:', domain_accuracy)