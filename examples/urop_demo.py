from medinify.sentiment import ReviewClassifier
from nltk import RegexpTokenizer
from nltk.corpus import stopwords

def main(classifier_type, model_input_file):
    """Classifies a single comments
    """

    classifier = ReviewClassifier(classifier_type=classifier_type)
    classifier.load_model(model_input_file)

    while True:
        comment = input('Write a review: \n')
        if comment == 'exit':
            break

        # Make lowercase
        lower_comment = comment.lower()

        # Remove punctuation and tokenize
        tokenizer = RegexpTokenizer(r'\w+')
        word_tokens = tokenizer.tokenize(lower_comment)

        # Remove stopwords and transform into BOW representation
        stop_words = set(stopwords.words('english'))
        filtered_comment = {
            word: True for word in word_tokens if word not in stop_words
        }

        if classifier_type == 'nb':
            print('\n' + classifier.model.classify(filtered_comment) + " :: " + comment + '\n')

        else:
            if classifier_type in ['rf', 'svm']:
                vectorized_comment = classifier.vectorizer.transform(filtered_comment)
                predict_output = classifier.model.predict(vectorized_comment)
                sentiment = ''
                if predict_output[0] == [0]:
                    sentiment = 'neg'
                elif predict_output[0] == [1]:
                    sentiment = 'pos'
                print('\n' + sentiment + ' :: ' + comment + '\n')
            elif classifier.classifier_type == 'nn':
                vectorized_comment = classifier.vectorizer.transform(filtered_comment)
                predict_output = classifier.model.predict_classes(vectorized_comment)
                sentiment = ''
                if predict_output[0] == 0:
                    sentiment = 'neg'
                elif predict_output[0] == 1:
                    sentiment = 'pos'
                print('\n' + sentiment + ' :: ' + comment + '\n')


if __name__ == '__main__':
    main(classifier_type='svm', model_input_file='examples/trained_svm_model.pickle')
    # change file name to change model
