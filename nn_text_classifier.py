import numpy
from keras.models import Sequential
from keras.layers import Dense, Dropout
# from gensim.models import word2vec
from gensim.models.keyedvectors import KeyedVectors

import argparse
import numpy as np
import csv
import string
import math
import random

from keras.utils import plot_model
from sklearn.model_selection import StratifiedKFold

np.random.seed(123)


def format_sentence(sent, swords=None):
    # convert to lowercase
    sent = sent.translate(str.maketrans("", "", string.punctuation)).lower()
    # remove stopwords
    if swords is not None:
        com_list = sent.split()
        filtered_words = []
        for word in com_list:
            if word not in swords and word in w2v_model.wv.vocab and len(w2v_model[word].tolist()) != 0:
                filtered_words.append(w2v_model[word].tolist())
    else:
        com_list = sent.split()
        filtered_words = []
        for word in com_list:
            if word in w2v_model.wv.vocab and len(w2v_model[word].tolist()) != 0:
                filtered_words.append(w2v_model[word].tolist())

    # Sum all of the word vectors
    # comment = [0.0 for _ in range(w2vsize)]
    # for word in filtered_words:
    #     for i in range(w2vsize):
    #         comment[i] += word[i]

    # Average all of the word vectors
    comment = [0.0 for _ in range(w2vsize)]
    for word in filtered_words:
        for i in range(w2vsize):
            comment[i] += word[i]
    if len(filtered_words) > 0:
        for i in range(w2vsize):
            comment[i] = comment[i]/len(filtered_words)

    return comment


# Import csv file
def import_csv(filename):
    my_list = []
    with open(filename) as commentfile:
        reader = csv.DictReader(commentfile)
        for row in reader:
            my_list.append({'comment': row['comment'], 'rating': row['rating']})

    # Parse and convert positive and negative examples.
    pos_words = []
    neg_words = []
    for c_dict in my_list:
        tmp_com = c_dict['comment']
        tmp_rating = c_dict['rating']

        # remove stop words
        with open(args.s) as c_raw:
            c_stopwords = c_raw.read().translate(str.maketrans("", "", string.punctuation)).splitlines()

            if float(tmp_rating) <= float(args.n):
                neg_words.append((format_sentence(tmp_com, c_stopwords), 0))
            if float(tmp_rating) >= float(args.p):
                pos_words.append((format_sentence(tmp_com, c_stopwords), 1))

    # neg_words.extend(neg_words)
    # neg_words.extend(neg_words[:100])
    print("Total Negative Instances:" + str(len(neg_words)) + "\nTotal Positive Instances:" + str(len(pos_words)))
    return neg_words, pos_words


# Parse input arguments
parser = argparse.ArgumentParser(description='Train a DNN Sentiment Classifier')
parser.add_argument('-i', metavar='inputfile', type=str,
                    help='path to the input csv file for training and testing.', required=True)
parser.add_argument('-c', metavar='toclassify', type=str, help='path to file with entries to classify.',
                    required=False, default=None)
parser.add_argument('-s', metavar='stopwords', type=str, help='path to stopwords file', required=True)
parser.add_argument('-p', metavar='posratings', type=str, help='a list of positive ratings as strings',
                    required=False, default=['4', '5'])
parser.add_argument('-n', metavar='negratings', type=str, help='a list of negative ratings as strings',
                    required=False, default=['1', '2'])
parser.add_argument('-z', metavar='iterations', type=str,
                    help='the number of times to repeat the classifier training', required=False, default=2)
parser.add_argument('-d', metavar='domain', type=str, help='a file with text from a different domain.',
                    required=False, default=None)
parser.add_argument('-m', metavar='model', type=str, help='location of word2vec model to be used',
                    required=False, default='/media/sf_Grad_School/GoogleNews-vectors-negative300.bin')

args = parser.parse_args()

# Build word2vec model
w2vsize = 300
# print("Building word2vec model on {}".format(args.i))
# sentences = word2vec.Text8Corpus(args.i)
# w2v_model = word2vec.Word2Vec(sentences, size=w2vsize, min_count=1, workers=4)
# You can get google's word2vec model here:
# https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing
print("word2vec model {}".format(args.m))
w2v_model = KeyedVectors.load_word2vec_format(args.m, binary=True)
neg_list, pos_list = import_csv(args.i)
# print(neg_list)
seed = 123
numpy.random.seed(seed)
# dataset = neg_list.extend(pos_list)
# print(dataset)
# X = pos_list[:0]
# print(X)
# Y = pos_list[:1]
# print(Y)
negcutoff = math.floor(len(neg_list) * 1)
poscutoff = math.floor(len(pos_list) * 1)
neg_idx_train = sorted(random.sample(range(len(neg_list)), negcutoff))
neg_train = [neg_list[i] for i in neg_idx_train]

pos_idx_train = sorted(random.sample(range(len(pos_list)), poscutoff))
pos_train = [pos_list[i] for i in pos_idx_train]

train = neg_train + pos_train

X = numpy.array([x[0] for x in train])
Y = numpy.array([x[1] for x in train])


kfold = StratifiedKFold(n_splits=int(args.z), shuffle=True, random_state=seed)
cvscores = []
for train, test in kfold.split(X,Y):
    model = Sequential()
    model.add(Dense(w2vsize, input_shape=(w2vsize, ), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(w2vsize, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(w2vsize*2, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(w2vsize, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X[train], Y[train], epochs=50, batch_size=20, verbose=0)
    scores = model.evaluate(X[test], Y[test], verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)
    # plot_model(model, to_file='model.png')

print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))

# create training and test sets
# set the cutoffs
# negcutoff = math.floor(len(neg_list) * 3 / 4)
# poscutoff = math.floor(len(pos_list) * 3 / 4)
#
# avgAccuracy = 0
# print("Training {} times...".format(args.z))
# for z in range(int(args.z)):
#     # train = neg_list[:negcutoff] + pos_list[:poscutoff]
#     # test = neg_list[negcutoff:] + pos_list[poscutoff:]
#     neg_idx_train = sorted(random.sample(range(len(neg_list)), negcutoff))
#     neg_train = [neg_list[i] for i in neg_idx_train]
#
#     neg_idx_test = set(range(len(neg_list))) - set(neg_idx_train)
#     neg_test = [neg_list[i] for i in neg_idx_test]
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
#     train_data = [x[0] for x in train]
#     train_labels = [x[1] for x in train]
#     test_data = [x[0] for x in test]
#     test_labels = [x[1] for x in test]
#     model.fit(train_data, train_labels, epochs=20, batch_size=10)
#     scores = model.evaluate(test_data, test_labels)
#     avgAccuracy += scores[1]
#     print("Test data accuracy: {}".format(scores[1]*100))
#     # print("{}: {}".format(model.metrics_names[1], scores[1]*100))
#
# print("Average Accuracy: " + str(avgAccuracy / int(args.z)))

# Import the file needing classification.
if args.c is not None:
    model = Sequential()
    model.add(Dense(w2vsize, input_shape=(w2vsize,), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(w2vsize, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(w2vsize * 2, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(w2vsize, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X, Y, epochs=50, batch_size=20, verbose=0)

    with open(args.c) as file:
        toclass = file.readlines()
    print("Predicting...")
    predict_this = []
    sentences = []
    for phrase in toclass:
        predict_this.append(format_sentence(phrase))
        sentences.append(phrase)
    answers = []
    predictions = model.predict(predict_this)
    for i in range(len(predictions)):
        answers.append([predictions[i], sentences[i]])
        print("{} :: {}".format(predictions[i], sentences[i]))


if args.d is not None:
    domain_list = []
    with open(args.d) as domainfile:
        d_reader = csv.DictReader(domainfile)
        for d_row in d_reader:
            domain_list.append({'comment': d_row['comment'], 'rating': d_row['rating']})
    print("{} length: {}".format(args.d, len(domain_list)))

    d_list = []
    for c in range(len(domain_list)):
        tmp_c = domain_list[c]['comment']
        tmp_r = domain_list[c]['rating']
        # remove stop words
        with open(args.s) as raw:
            stopwords = raw.read().translate(str.maketrans("", "", string.punctuation)).splitlines()
            if tmp_r in args.n:
                d_list.append((format_sentence(tmp_c, stopwords), 0))
            if tmp_r in args.p:
                d_list.append((format_sentence(tmp_c, stopwords), 1))

    d_data = [word[0] for word in d_list]
    d_labels = [word[1] for word in d_list]

    model = Sequential()
    model.add(Dense(w2vsize, input_shape=(w2vsize,), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(w2vsize, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(w2vsize * 2, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(w2vsize, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X, Y, epochs=50, batch_size=20, verbose=0)

    domain_accuracy = model.evaluate(d_data, d_labels)
    print("Classifier domain shift accuracy: {}".format(domain_accuracy[1]*100))
    # print("\n{}: {}".format(model.metrics_names[1], domain_accuracy[1]*100))