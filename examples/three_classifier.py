# Medinify classifier to sort reviews into negative, neutral and positive reviews
from medinify.sentiment import ReviewClassifier
from medinify.sentiment.rnn_review_classifier import RNNReviewClassifier
from medinify.sentiment.rnn_review_classifier import RNNNetwork
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
    """
    plist=[]
    three_classifier = ReviewClassifier("rnn", 5, poslist=plist)
    data3, target3 = three_classifier.preprocess('diabetes.csv')
    three_classifier.fit(data3, target3)
    three_classifier.evaluate_average_accuracy('diabetes.csv', 10)
    """
    #three_classifier.classify('classify.txt', csv_file='diabetes.csv', evaluate=True)
    """
    var3, var4 = three_classifier.evaluate(three_network, dataloadervalid)
    print(str(var) + " " + str(var2 * 100) + "%")
    print(str(var3) + " " + str(var4 * 100) + "%")
    """
    INPUT_DIM = 25000
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 256
    OUTPUT_DIM = 1
    N_LAYERS = 2
    BIDIRECTIONAL = True
    DROPOUT = 0.5
    
    three_classifier = RNNReviewClassifier()
    dataloadertrain, dataloadervalid, padid, unkid = three_classifier.get_data_loaders('diabetes.csv', 'heart.csv', 64)
    three_network = RNNNetwork(INPUT_DIM, 
            EMBEDDING_DIM, 
            HIDDEN_DIM, 
            OUTPUT_DIM, 
            N_LAYERS, 
            BIDIRECTIONAL, 
            DROPOUT, padid, unkid)
    
    three_classifier.train(three_network, dataloadertrain)
    three_classifier.evaluate2(three_network, dataloadervalid)
    
if __name__ == "__main__":
    main()
