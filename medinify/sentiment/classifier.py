
import numpy as np
import pickle
from medinify.datasets.dataset import Dataset
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold


class Classifier:
    """
    The classifier class implements three SkLearn-Based sentiment classifiers
    MultinomialNaiveBayes, Random Forest, and SVC
    For training, evaluation, and validation (k-fold)

    Attributes:
        classifier_type: what type of classifier to train ('nb', 'rf', or 'svm')
        count_vectorizer: turns strings into count vectors
        tfidf_vectorizer: turns strings into tfidf vectors
        pos_count_vectorizer: turns string into count vectors for a certain part of speech
        processor: processes data and ratings into numeric representations
        data_representation: how comment data should be numerically represented
            ('count', 'tfidf', 'embedding', or 'pos')
        num_classes: number of rating classes
        rating_type: type of rating to use if multiple exist in the dataset
        pos_threshold: postive rating threshold
        neg_threshold: negative rating threshold
        w2v_file: path to w2v file if using averaged embeddings
        pos: part of speech if using pos count vectors
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

    def fit(self, reviews_file=None, data=None, target=None):
        """
        Trains a model on review data
        :param reviews_file: path to csv containing training data
        :param data: data ndarray
        :param target: target ndarray
        :return: a trained model
        """
        if bool(reviews_file):
            data, target = self.load_data(reviews_file)

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

    def evaluate(self, trained_model, eval_reviews_csv=None, data=None, target=None):
        """
        Evaluates the accuracy, precision, recall, and F-1 score
        of a trained model over a review dataset
        :param trained_model: trained model
        :param eval_reviews_csv: path to csv of review data being evaluated
        :param data: ndarray of data
        :param target: ndarray of target
        :return eval_metrics: calculated evaluation metrics (accuracy, precision, recall, f_measure)
        """
        if eval_reviews_csv:
            data, target = self.load_data(eval_reviews_csv)
        predictions = trained_model.predict(data)
        accuracy = accuracy_score(target, predictions)
        precisions = precision_score(target, predictions, average=None)
        recalls = recall_score(target, predictions, average=None)
        f_scores = f1_score(target, predictions, average=None)
        matrix = confusion_matrix(target, predictions)

        eval_metrics = {}

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
            print('Confusion Matrix:')
            print(matrix)

            eval_metrics = {'accuracy': accuracy, 'pos_precision': pos_precision,
                            'pos_recall': pos_recall, 'neg_precision': neg_precision,
                            'neg_recall': neg_recall, 'pos_f_measure': pos_f_measure,
                            'neg_f_measure': neg_f_measure}

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
            print('Confusion Matrix:')
            print(matrix)

            eval_metrics = {'accuracy': accuracy, 'pos_precision': pos_precision,
                            'pos_recall': pos_recall, 'pos_f_measure': pos_f_measure,
                            'neutral_precision': neutral_precision, 'neutral_recall': neutral_recall,
                            'neutral_f_measure': neutral_f_measure, 'neg_precision': neg_precision,
                            'neg_recall': neg_recall, 'neg_f_measure': neg_f_measure}

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

            eval_metrics = {'accuracy': accuracy,
                            'one_star_precision': one_star_precision,
                            'one_star_recall': one_star_recall,
                            'one_star_f_measure': one_star_f_measure,
                            'two_star_precision': two_star_precision,
                            'two_star_recall': two_star_recall,
                            'two_star_f_measure': two_star_f_measure,
                            'three_star_precision': three_star_precision,
                            'three_star_recall': three_star_recall,
                            'three_star_f_measure': three_star_f_measure,
                            'four_star_precision': four_star_precision,
                            'four_star_recall': four_star_recall,
                            'four_star_f_measure': four_star_f_measure,
                            'five_star_precision': five_star_precision,
                            'five_star_recall': five_star_recall,
                            'five_star_f_measure': five_star_f_measure}

        return eval_metrics

    def validate(self, review_csv, k_folds=10):
        """
        Runs k-fold cross validation
        :param k_folds: number of folds
        :param review_csv: csv with data to splits, train, and validate on
        """
        data, target = self.load_data(review_csv)
        skf = StratifiedKFold(n_splits=k_folds)

        accuracies = []
        if self.num_classes == 2:
            pos_precisions = []
            neg_precisions = []
            pos_recalls = []
            neg_recalls = []
            pos_f_scores = []
            neg_f_scores = []
        elif self.num_classes == 3:
            pos_precisions = []
            neutral_precisions = []
            neg_precisions = []
            pos_recalls = []
            neutral_recalls = []
            neg_recalls = []
            pos_f_scores = []
            neutral_f_scores = []
            neg_f_scores = []
        elif self.num_classes == 5:
            one_star_precisions = []
            one_star_recalls = []
            one_star_f_scores = []
            two_star_precisions = []
            two_star_recalls = []
            two_star_f_scores = []
            three_star_precisions = []
            three_star_recalls = []
            three_star_f_scores = []
            four_star_precisions = []
            four_star_recalls = []
            four_star_f_scores = []
            five_star_precisions = []
            five_star_recalls = []
            five_star_f_scores = []

        num_fold = 1
        for train, test in skf.split(data, target):
            train_data = np.asarray([data[x] for x in train])
            train_target = np.asarray([target[x] for x in train])
            test_data = np.asarray([data[x] for x in test])
            test_target = np.asarray([target[x] for x in test])

            model = self.fit(data=train_data, target=train_target)
            print('Fold {}:'.format(num_fold))
            eval_metrics = self.evaluate(model, data=test_data, target=test_target)

            if self.num_classes == 2:
                accuracies.append(eval_metrics['accuracy'])
                pos_precisions.append(eval_metrics['pos_precision'])
                neg_precisions.append(eval_metrics['neg_precision'])
                pos_recalls.append(eval_metrics['pos_recall'])
                neg_recalls.append(eval_metrics['neg_recall'])
                pos_f_scores.append(eval_metrics['pos_f_measure'])
                neg_f_scores.append(eval_metrics['neg_f_measure'])
            elif self.num_classes == 3:
                accuracies.append(eval_metrics['accuracy'])
                pos_precisions.append(eval_metrics['pos_precision'])
                neutral_precisions.append(eval_metrics['neutral_precision'])
                neg_precisions.append(eval_metrics['neg_precision'])
                pos_recalls.append(eval_metrics['pos_recall'])
                neutral_recalls.append(eval_metrics['neutral_recall'])
                neg_recalls.append(eval_metrics['neg_recall'])
                pos_f_scores.append(eval_metrics['pos_f_measure'])
                neutral_f_scores.append(eval_metrics['neutral_f_measure'])
                neg_f_scores.append(eval_metrics['neg_f_measure'])
            elif self.num_classes == 5:
                accuracies.append(eval_metrics['accuracy'])
                one_star_precisions.append(eval_metrics['one_star_precision'])
                one_star_recalls.append(eval_metrics['one_star_recall'])
                one_star_f_scores.append(eval_metrics['one_star_f_measure'])
                two_star_precisions.append(eval_metrics['two_star_precision'])
                two_star_recalls.append(eval_metrics['two_star_recall'])
                two_star_f_scores.append(eval_metrics['two_star_f_measure'])
                three_star_precisions.append(eval_metrics['three_star_precision'])
                three_star_recalls.append(eval_metrics['three_star_recall'])
                three_star_f_scores.append(eval_metrics['three_star_f_measure'])
                four_star_precisions.append(eval_metrics['four_star_precision'])
                four_star_recalls.append(eval_metrics['four_star_recall'])
                four_star_f_scores.append(eval_metrics['four_star_f_measure'])
                five_star_precisions.append(eval_metrics['five_star_precision'])
                five_star_recalls.append(eval_metrics['five_star_recall'])
                five_star_f_scores.append(eval_metrics['five_star_f_measure'])

            num_fold += 1

        average_accuracy = np.mean(accuracies)
        accuracy_std = np.std(accuracies)

        if self.num_classes == 2:
            average_pos_precision = np.mean(pos_precisions)
            average_neg_precision = np.mean(neg_precisions)
            average_pos_recall = np.mean(pos_recalls)
            average_neg_recall = np.mean(neg_recalls)
            average_pos_f_measure = np.mean(pos_f_scores)
            average_neg_f_measure = np.mean(neg_f_scores)
            pos_precision_std = np.std(pos_precisions)
            neg_precision_std = np.std(neg_precisions)
            pos_recall_std = np.std(pos_recalls)
            neg_recall_std = np.std(neg_recalls)
            pos_f_measure_std = np.std(pos_f_scores)
            neg_f_measure_std = np.std(neg_f_scores)

            print('\nValidation Metrics:')
            print('Average Accuracy: {:.4f}% +/-{:.4f}%'.format(
                average_accuracy * 100, accuracy_std * 100))
            print('Average Pos Precision: {:.4f}% +/-{:.4f}%'.format(
                average_pos_precision * 100, pos_precision_std * 100))
            print('Average Pos Recall: {:.4f}% +/-{:.4f}%'.format(
                average_pos_recall * 100, pos_recall_std * 100))
            print('Average Pos F-Measure: {:.4f}% +/-{:.4f}%'.format(
                average_pos_f_measure * 100, pos_f_measure_std * 100))
            print('Average Neg Precision: {:.4f}% +/-{:.4f}%'.format(
                average_neg_precision * 100, neg_precision_std * 100))
            print('Average Neg Recall: {:.4f}% +/-{:.4f}%'.format(
                average_neg_recall * 100, neg_recall_std * 100))
            print('Average Neg F-Measure: {:.4f}% +/-{:.4f}%'.format(
                average_neg_f_measure * 100, neg_f_measure_std * 100))
        elif self.num_classes == 3:
            average_pos_precision = np.mean(pos_precisions)
            average_neutral_precision = np.mean(neutral_precisions)
            average_neg_precision = np.mean(neg_precisions)
            average_pos_recall = np.mean(pos_recalls)
            average_neutral_recall = np.mean(neutral_recalls)
            average_neg_recall = np.mean(neg_recalls)
            average_pos_f_measure = np.mean(pos_f_scores)
            average_neutral_f_measure = np.mean(neutral_f_scores)
            average_neg_f_measure = np.mean(neg_f_scores)
            pos_precision_std = np.std(pos_precisions)
            neutral_precision_std = np.std(neutral_precisions)
            neg_precision_std = np.std(neg_precisions)
            pos_recall_std = np.std(pos_recalls)
            neutral_recall_std = np.std(neutral_recalls)
            neg_recall_std = np.std(neg_recalls)
            pos_f_measure_std = np.std(pos_f_scores)
            neutral_f_measure_std = np.std(neutral_f_scores)
            neg_f_measure_std = np.std(neg_f_scores)

            print('\nValidation Metrics:')
            print('Average Accuracy: {:.4f}% +/-{:.4f}%'.format(
                average_accuracy * 100, accuracy_std * 100))
            print('Average Pos Precision: {:.4f}% +/-{:.4f}%'.format(
                average_pos_precision * 100, pos_precision_std * 100))
            print('Average Pos Recall: {:.4f}% +/-{:.4f}%'.format(
                average_pos_recall * 100, pos_recall_std * 100))
            print('Average Pos F-Measure: {:.4f}% +/-{:.4f}%'.format(
                average_pos_f_measure * 100, pos_f_measure_std * 100))
            print('Average Neutral Precision: {:.4f}% +/-{:.4f}%'.format(
                average_neutral_precision * 100, neutral_precision_std * 100))
            print('Average Neutral Recall: {:.4f}% +/-{:.4f}%'.format(
                average_neutral_recall * 100, neutral_recall_std * 100))
            print('Average Neutral F-Measure: {:.4f}% +/-{:.4f}%'.format(
                average_neutral_f_measure * 100, neutral_f_measure_std * 100))
            print('Average Neg Precision: {:.4f}% +/-{:.4f}%'.format(
                average_neg_precision * 100, neg_precision_std * 100))
            print('Average Neg Recall: {:.4f}% +/-{:.4f}%'.format(
                average_neg_recall * 100, neg_recall_std * 100))
            print('Average Neg F-Measure: {:.4f}% +/-{:.4f}%'.format(
                average_neg_f_measure * 100, neg_f_measure_std * 100))
        elif self.num_classes == 5:
            average_one_star_precision = np.mean(one_star_precisions)
            one_star_precision_std = np.std(one_star_precisions)
            average_one_star_recall = np.mean(one_star_recalls)
            one_star_recall_std = np.std(one_star_recalls)
            average_one_star_f_measure = np.mean(one_star_f_scores)
            one_star_f_measure_std = np.std(one_star_f_scores)

            average_two_star_precision = np.mean(two_star_precisions)
            two_star_precision_std = np.std(two_star_precisions)
            average_two_star_recall = np.mean(two_star_recalls)
            two_star_recall_std = np.std(two_star_recalls)
            average_two_star_f_measure = np.mean(two_star_f_scores)
            two_star_f_measure_std = np.std(two_star_f_scores)

            average_three_star_precision = np.mean(three_star_precisions)
            three_star_precision_std = np.std(three_star_precisions)
            average_three_star_recall = np.mean(three_star_recalls)
            three_star_recall_std = np.std(three_star_recalls)
            average_three_star_f_measure = np.mean(three_star_f_scores)
            three_star_f_measure_std = np.std(three_star_f_scores)

            average_four_star_precision = np.mean(four_star_precisions)
            four_star_precision_std = np.std(four_star_precisions)
            average_four_star_recall = np.mean(four_star_recalls)
            four_star_recall_std = np.std(four_star_recalls)
            average_four_star_f_measure = np.mean(four_star_f_scores)
            four_star_f_measure_std = np.std(four_star_f_scores)

            average_five_star_precision = np.mean(five_star_precisions)
            five_star_precision_std = np.std(five_star_precisions)
            average_five_star_recall = np.mean(five_star_recalls)
            five_star_recall_std = np.std(five_star_recalls)
            average_five_star_f_measure = np.mean(five_star_f_scores)
            five_star_f_measure_std = np.std(five_star_f_scores)

            print('\nValidation Metrics:')
            print('Average Accuracy: {:.4f}% +/-{:.4f}%'.format(
                average_accuracy * 100, accuracy_std * 100))

            print('Average One Star Precision: {:.4f}% +/-{:.4f}%'.format(
                average_one_star_precision * 100, one_star_precision_std * 100))
            print('Average One Star Recall: {:.4f}% +/-{:.4f}%'.format(
                average_one_star_recall * 100, one_star_recall_std * 100))
            print('Average One Star F-Measure: {:.4f}% +/-{:.4f}%'.format(
                average_one_star_f_measure * 100, one_star_f_measure_std * 100))

            print('Average Two Star Precision: {:.4f}% +/-{:.4f}%'.format(
                average_two_star_precision * 100, two_star_precision_std * 100))
            print('Average Two Star Recall: {:.4f}% +/-{:.4f}%'.format(
                average_two_star_recall * 100, two_star_recall_std * 100))
            print('Average Two Star F-Measure: {:.4f}% +/-{:.4f}%'.format(
                average_two_star_f_measure * 100, two_star_f_measure_std * 100))

            print('Average Three Star Precision: {:.4f}% +/-{:.4f}%'.format(
                average_three_star_precision * 100, three_star_precision_std * 100))
            print('Average Three Star Recall: {:.4f}% +/-{:.4f}%'.format(
                average_three_star_recall * 100, three_star_recall_std * 100))
            print('Average Three Star F-Measure: {:.4f}% +/-{:.4f}%'.format(
                average_three_star_f_measure * 100, three_star_f_measure_std * 100))

            print('Average Four Star Precision: {:.4f}% +/-{:.4f}%'.format(
                average_four_star_precision * 100, four_star_precision_std * 100))
            print('Average Four Star Recall: {:.4f}% +/-{:.4f}%'.format(
                average_four_star_recall * 100, four_star_recall_std * 100))
            print('Average Four Star F-Measure: {:.4f}% +/-{:.4f}%'.format(
                average_four_star_f_measure * 100, four_star_f_measure_std * 100))

            print('Average Five Star Precision: {:.4f}% +/-{:.4f}%'.format(
                average_five_star_precision * 100, five_star_precision_std * 100))
            print('Average Five Star Recall: {:.4f}% +/-{:.4f}%'.format(
                average_five_star_recall * 100, five_star_recall_std * 100))
            print('Average Five Star F-Measure: {:.4f}% +/-{:.4f}%'.format(
                average_five_star_f_measure * 100, five_star_f_measure_std * 100))

    def save_model(self, trained_model, output_file):
        """
        Saves a trained model and its processor
        :param trained_model: trained model
        :param output_file: path to output saved model file
        """
        with open(output_file, 'wb') as f:
            pickle.dump(trained_model, f, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.processor, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_model(self, model_file):
        """
        Loads model and processor from pickled format
        :param model_file: path to pickled model file
        :return: loaded model
        """
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
            self.processor = pickle.load(f)
        return model

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
