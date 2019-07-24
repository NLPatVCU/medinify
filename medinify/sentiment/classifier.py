
from medinify.datasets.dataset import Dataset
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix


class Classifier:
    """
    The classifier class implements three SkLearn-Based sentiment classifiers
    MultinomialNaiveBayes, Random Forest, and SVC
    For training, evaluation, and validation (k-fold)
    """

    classifier_type = None
    count_vectorizer = None
    tfidf_vectorizer = None
    pos_count_vectorizer = None
    processor = None

    def __init__(self, classifier_type=None, data_representation='count',
                 num_classes=2, rating_type='effectiveness',
                 pos_threshold=4.0, neg_threshold=2.0, w2v_file=None, pos=None):
        assert classifier_type in ['nb', 'rf', 'svm'], 'Classifier Type must be \'nb\', \'rf\', or \'svm\''
        self.classifier_type = classifier_type
        self.data_representation = data_representation
        self.num_classes = num_classes
        self.rating_type = rating_type
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold
        self.w2v_file = w2v_file
        self.pos = pos

    def fit(self, review_csv):
        """
        Trains a model on review data
        :param review_csv: path to csv containing training data
        :return: a trained model
        """
        data, target = self.load_data(review_csv)

        model = None
        if self.classifier_type == 'nb':
            model = MultinomialNB()
        elif self.classifier_type == 'rf':
            model = RandomForestClassifier(n_estimators=100)
        elif self.classifier_type == 'svm':
            model = SVC(kernel='rbf', C=10, gamma=0.01)

        print('Fitting model...')
        model.fit(data, target)
        print('Model fit.')
        return model

    def evaluate(self, trained_model, eval_reviews_csv):
        """
        Evaluates the accuracy, precision, recall, and F-1 score
        of a trained model over a review dataset
        :param trained_model: trained model
        :param eval_reviews_csv: path to csv of review data being evaluated
        """
        data, target = self.load_data(eval_reviews_csv)
        predictions = trained_model.predict(data)
        accuracy = accuracy_score(target, predictions)
        precisions = precision_score(target, predictions, average=None)
        recalls = recall_score(target, predictions, average=None)
        f_scores = f1_score(target, predictions, average=None)
        matrix = confusion_matrix(target, predictions)

        print('\nEvaluation Metrics:\n')
        print('Accuracy: {:.4f}%'.format(accuracy * 100))

        if self.num_classes == 2:
            neg_precision = precisions[0]
            pos_precision = precisions[1]
            neg_recall = recalls[0]
            pos_recall = recalls[1]
            neg_f_measure = f_scores[0]
            pos_f_measure = f_scores[1]

            print('Positive Precision: {:.4f}%'.format(pos_precision * 100))
            print('Positive Recall: {:.4f}%'.format(pos_recall * 100))
            print('Positive F-Score: {:.4f}%'.format(pos_f_measure * 100))
            print('Negative Precision: {:.4f}%'.format(neg_precision * 100))
            print('Negative Recall: {:.4f}%'.format(neg_recall * 100))
            print('Negative F-Score: {:.4f}%'.format(neg_f_measure * 100))

        elif self.num_classes == 3:
            neg_precision = precisions[0]
            neutral_precision = precisions[1]
            pos_precision = precisions[2]
            neg_recall = recalls[0]
            neutral_recall = recalls[1]
            pos_recall = recalls[2]
            neg_f_measure = f_scores[0]
            neutral_f_measure = f_scores[1]
            pos_f_measure = f_scores[2]

            print('Positive Precision: {:.4f}%'.format(pos_precision * 100))
            print('Positive Recall: {:.4f}%'.format(pos_recall * 100))
            print('Positive F-Score: {:.4f}%'.format(pos_f_measure * 100))
            print('Neutral Precision: {:.4f}%'.format(neutral_precision * 100))
            print('Neutral Recall: {:.4f}%'.format(neutral_recall * 100))
            print('Neutral F-Score: {:.4f}%'.format(neutral_f_measure * 100))
            print('Negative Precision: {:.4f}%'.format(neg_precision * 100))
            print('Negative Recall: {:.4f}%'.format(neg_recall * 100))
            print('Negative F-Score: {:.4f}%'.format(neg_f_measure * 100))

        elif self.num_classes == 5:
            one_star_precision = precisions[0]
            two_star_precision = precisions[1]
            three_star_precision = precisions[2]
            four_star_precision = precisions[3]
            five_star_precision = precisions[4]
            one_star_recall = recalls[0]
            two_star_recall = recalls[1]
            three_star_recall = recalls[2]
            four_star_recall = recalls[3]
            five_star_recall = recalls[4]
            one_star_f_measure = f_scores[0]
            two_star_f_measure = f_scores[1]
            three_star_f_measure = f_scores[2]
            four_star_f_measure = f_scores[3]
            five_star_f_measure = f_scores[4]

            print('One Star Precision: {:.4f}%'.format(one_star_precision * 100))
            print('One Star Recall: {:.4f}%'.format(one_star_recall * 100))
            print('One Star F-Score: {:.4f}%'.format(one_star_f_measure * 100))
            print('Two Star Precision: {:.4f}%'.format(two_star_precision * 100))
            print('Two Star Recall: {:.4f}%'.format(two_star_recall * 100))
            print('Two Star F-Score: {:.4f}%'.format(two_star_f_measure * 100))
            print('Three Star Precision: {:.4f}%'.format(three_star_precision * 100))
            print('Three Star Recall: {:.4f}%'.format(three_star_recall * 100))
            print('Three Star F-Score: {:.4f}%'.format(three_star_f_measure * 100))
            print('Four Star Precision: {:.4f}%'.format(four_star_precision * 100))
            print('Four Star Recall: {:.4f}%'.format(four_star_recall * 100))
            print('Four Star F-Score: {:.4f}%'.format(four_star_f_measure * 100))
            print('Five Star Precision: {:.4f}%'.format(five_star_precision * 100))
            print('Five Star Recall: {:.4f}%'.format(five_star_recall * 100))
            print('Five Star F-Score: {:.4f}%'.format(five_star_f_measure * 100))

        print('Confusion Matrix:')
        print(matrix)

    def load_data(self, review_csv):
        """
        Loads and processes data from csv
        :param review_csv: path to csv with review data
        :return: data, target
        """
        dataset = Dataset(num_classes=self.num_classes,
                          rating_type=self.rating_type,
                          pos_threshold=self.pos_threshold,
                          neg_threshold=self.neg_threshold)

        dataset.load_file(review_csv)
        if self.processor:
            dataset.processor = self.processor

        data, target = None, None
        if self.data_representation == 'count':
            data, target = dataset.get_count_vectors()
        elif self.data_representation == 'tfidf':
            data, target = dataset.get_tfidf_vectors()
        elif self.data_representation == 'embedding':
            data, target = dataset.get_average_embeddings(w2v_file=self.w2v_file)
        elif self.data_representation == 'pos':
            data, target = dataset.get_pos_vectors(pos=self.pos)

        if not self.processor:
            self.processor = dataset.processor

        return data, target
