# Medinify

Medical text extraction and classification 

## Requirements

* Python 3.6
* Cannot use Python 3.7 until TensorFlow supports it (12/2/18)

## Getting Started

To install medinify, run:

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## review_classifier.py

Train and test eiher a Naive Bayes Classifier to classify drug reviews.

### Examples

After installing medinify, run:

```python
from medinify.sentiment import ReviewClassifier

review_classifier = ReviewClassifier('nb', 'stopwords.txt')
review_classifier.train('citalopram_train.csv')
review_classifier.classify('neutral.txt')
```

### Example

After installing medinify, run:

```python
from medinify.scrapers import WebMDScraper

input_url = "https://www.webmd.com/drugs/drugreview-1701-citalopram-oral.aspx?drugid=1701&drugname=citalopram-oral"
scraper = WebMDScraper("citalopram_train.csv")
scraper.scrape(input_url, 10)

```

## Contributions

1. Copy the URL from the Medinify repository and use Git to
clone the repo:

```bash
# Clone the repo into current directory
git clone https://github.com/NanoNLP/medinify.git
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

7. Finally, push the changes to the current branch of your forked repository:

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