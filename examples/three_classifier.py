# Medinify classifier to sort reviews into negative, neutral and positive reviews
from medinify.sentiment import ReviewClassifier
from medinify.datasets import ReviewDataset
import sys

def main(): 
    """
    dataset = ReviewDataset('drug-name', 'WebMD')
    dataset.load('/home/ishaan/Desktop/Medinify/medinify/data/common-dataset-2019-03-02.pickle')
    dataset.generate_ratings()
    dataset.remove_empty_comments()
    dataset.write_file("csv", "cancer.csv")
    
    
    dataset2 = ReviewDataset('drug-name', 'WebMD')
    dataset2.load('/home/ishaan/Desktop/Medinify/medinify/data/heart-dataset-2019-03-05.pickle')
    dataset2.generate_ratings()
    dataset2.remove_empty_comments()
    dataset2.write_file("csv", "depression.csv")
    """

    three_classifier = ReviewClassifier("nb", 3)
    data3, target3 = three_classifier.preprocess('diabetes.csv')
    three_classifier.fit(data3, target3)
    three_classifier.evaluate_average_accuracy('diabetes.csv', 10)
    #three_classifier.classify('classify.txt', csv_file='diabetes.csv', evaluate=True)
if __name__ == "__main__":
    main()
