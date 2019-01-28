# medinify

Medical text extraction and classification 

## Requirements

* Python 3.6
* Cannot use Python 3.7 until TensorFlow supports it (12/2/18)

## Getting Started

To install medinify, run:

```bash
pip install git+https://github.com/NanoNLP/medinify.git
```

and after installation, run:

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## text_classifier.py

Train and test either a Naive Bayes Classifier (text_classifier), a Decision Tree classifier (dt_text_classifier), or a Neural Network (nn_text_classifier) to classify drug reviews.

### Text Classifier Options

**-i**  Required. Input CSV file that includes training and testing data. Must be in the format of "review text","5", where the second entry is the rating.  See the "citalopram_effectiveness.csv" file for an example.  The program divides this data up into 3/4 used for training, and 1/4 used for testing to calculate the accuracy.

**-s**  Required. Stopwords text file with a list of stopwords to remove before training the classifier or predicting sentiment class.  See the "stopwords_long.txt" file for an example.

**-c**  Optional, default = None. Input text file with one review per line that needs classification. Use this option to predict semtiment class on reviews that do not yet have a rating, or to polarize neutral reviews.  See the "neutral.txt" file for an example format.

**-d**  Optional, default = None. Input CSV file in the same format as the -i option.  This file contains additional ratings to classify and calculate accuracy.  This option is meant to analyze ratings from a different domain than the one being trained on.

**-p**  Optional, default = ['4','5']. A list of ratings that count as positive ratings for training the classifier.  These must be strings, and must match the ratings in the input files.

**-n** Optional, default = ['1','2']. A list of ratings that count as negative ratings for training the classifier.  These must be strings, and must match the ratings in the input files.

**-z**  Optional, default = 1.  The number of time to repeat training the classifier to get an average accuracy when choosing different training sets of data.

**-m** Required for Neural Network, not used for other NB or DT, defines the Word2Vec model to use.

### Examples

After installing medinify, run:

```bash
from medinify.sentiment.review_classifier import ReviewClassifier

def main():
   
    review_classifier = ReviewClassifier('nb', 'stopwords.txt')
    review_classifier.train('citalopram_train.csv')
    review_classifier.classify('neutral.txt')

if __name__ == "__main__":
    main()
```

## drug_review_scraper.py

Scrape WebMD for drug reviews.

### Drug Review Scraper Options

**-o** Required. Output file.

**-p** Optional, default=1. Number of pages to scrape.

**-i** Required. Input URL.

### Example

After installing medinify, run:

```bash
from medinify.scrapers import WebMDScraper

def main():

    input_url = "https://www.webmd.com/drugs/drugreview-1701-citalopram-oral.aspx?drugid=1701&drugname=citalopram-oral"
    scraper = WebMDScraper("citalopram_train.csv")
    scraper.scrape(input_url, 10)

if __name__ == "__main__":
    main()
```

Which extracts text reviews from the input URL and outputs them to

 ```bash 
 citalopram_train.csv
 ```

## Contributions

1. [Fork](http://help.github.com/fork-a-repo/) the project, clone your fork,
and configure the remotes:

```bash
# Clone your fork of the repo into the current directory
git clone https://github.com/<your-username>/medinify.git
# Navigate to the newly cloned directory
cd medinify
# Assign the original repo to a remote called "upstream"
git remote add upstream https://github.com/NanoNLP/medinify.git
```

2. If already cloned, update from the master:

```bash
git checkout master
git pull upstream master
```

3. Create a branch (off the main project development branch) to
contain your changes:

```bash 
git checkout -b <new-branch-name>
```

4. After making changes to files or adding new files to the project, add the changes to your local repository

```bash
git add -A
```

5. Next, we record the changes made and provide a message describing the changes made so others can understand

```bash
git commit -m "Description of changes made"
```

If many changes were made and the commit message is extensive, instead run 

```bash
git commit
```
and the deafault text editor will open for you to record a longer commit message.

6. After saving and exiting your commit message text file, verify what git will commit using the command:

```bash
git status
```

and you will recieve an output similar to this:

```bash
On branch <new-branch-name>
Your branch is ahead of 'origin/<new-branch-name>' by 1 commit.
  (use "git push" to publish your local commits)
nothing to commit, working directory clean
```

7. Finally, use the 
```bash
git push
``` 
command to push the changes to the current branch of your forked repository:

```bash
git push --set-upstream origin <new-branch-name>
```

### Making Pull Requests

After following the steps above, you can make a pull request to the original repository by navigating to your forked repository on github, and press the “New pull request” button on your left-hand side of the page.

Select the master branch from the left-hand side, and the desired branch of your forked repo from the right-hand side.

Then add a title, a comment, and then press the “Create pull request” button.

## More Info

- [VCU Natural Language Processing Lab](https://nlp.cs.vcu.edu/)     ![alt text](https://nlp.cs.vcu.edu/images/vcu_head_logo "VCU")
- [Nanoinformatics Vertically Integrated Projects](https://rampages.us/nanoinformatics/)