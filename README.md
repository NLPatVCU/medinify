# Medical Text Classifier
Text classification with a focus on medical text.

## Requirements
* Python 3.6
* Cannot use Python 3.7 until TensorFlow supports it (12/2/18)

## Installation
```
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## text_classifier.py
Train and test either a Naive Bayes Classifier (text_classifier), a Decision Tree classifier (dt_text_classifier), or a Neural Network (nn_text_classifier) to classify drug reviews.

### Options

**-i**  Required. Input CSV file that includes training and testing data. Must be in the format of "review text","5", where the second entry is the rating.  See the "citalopram_effectiveness.csv" file for an example.  The program divides this data up into 3/4 used for training, and 1/4 used for testing to calculate the accuracy.

**-s**  Required. Stopwords text file with a list of stopwords to remove before training the classifier or predicting sentiment class.  See the "stopwords_long.txt" file for an example.

**-c**  Optional, default = None. Input text file with one review per line that needs classification. Use this option to predict semtiment class on reviews that do not yet have a rating, or to polarize neutral reviews.  See the "neutral.txt" file for an example format.

**-d**  Optional, default = None. Input CSV file in the same format as the -i option.  This file contains additional ratings to classify and calculate accuracy.  This option is meant to analyze ratings from a different domain than the one being trained on.

**-p**  Optional, default = ['4','5']. A list of ratings that count as positive ratings for training the classifier.  These must be strings, and must match the ratings in the input files.

**-n** Optional, default = ['1','2']. A list of ratings that count as negative ratings for training the classifier.  These must be strings, and must match the ratings in the input files.

**-z**  Optional, default = 1.  The number of time to repeat training the classifier to get an average accuracy when choosing different training sets of data.

**-m** Required for Neural Network, not used for other NB or DT, defines the Word2Vec model to use.


### Examples
```
python text_classifier.py -i citalopram_train.csv -s stopwords.txt -c neutral.txt -z 10

python dt_text_classifier.py -i citalopram_train.csv -s stopwords.txt -c neutral.txt -z 10

python nn_text_classifier.py -i citalopram_train.csv -s stopwords.txt -c neutral.txt -d citalopram_effectivness.csv -p 4.0 -n 2.0 -z 10 -m GoogleNews-vectors-negative300.bin
 ```

## drug_review_scraper.py
Scrape WebMD for drug reviews.

### Options
**-o** Required. Output file.

**-p** Optional, default=1. Number of pages to scrape.

**-i** Required. Input URL.

### Example
```
python drug_review_scraper.py -i "https://www.webmd.com/drugs/drugreview-1701-citalopram-oral.aspx?drugid=1701&drugname=citalopram-oral" -o citalopram_train.csv -p 10
```