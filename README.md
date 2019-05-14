# Medinify

Medical text extraction and classification.

## Requirements

* Python 3.6
* Cannot use Python 3.7 until TensorFlow supports it (12/2/18)

## Getting Started

To install medinify and requirements in a virtual environment, run:

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## Datasets

### Review Datasets

Scrape for reviews from WebMD, save them, load them, print statistics, cleanse data, and export to CSV or JSON.

#### Review Dataset Examples

```python
from medinify.datasets import ReviewDataset

# For saving a Citalopram reviews dataset
review_dataset = ReviewDataset('Citalopram')
review_dataset.collect('https://www.webmd.com/drugs/drugreview-1701-citalopram-oral.aspx?drugid=1701&drugname=citalopram-oral')
review_dataset.save()
```

```python
from medinify.datasets import ReviewDataset

# Load a saved citalopram dataset, cleanse the data, write CSV, and print stats
review_dataset = ReviewDataset('Citalopram')
review_dataset.load()
review_dataset.generate_ratings()
review_dataset.write_file('csv')
review_dataset.print_stats()
```

## Classifiers

### Review Classifier

Train and test a model for running sentiment analysis on drug reviews. Models can currently use Naive Bayes, Decision Tree, or a Tensorflow Neural Network.

#### Review Classifier Examples

```python
from medinify.sentiment import ReviewClassifier
from medinify.sentiment import NeuralNetReviewClassifier

# Train a use a classifier if you already have a Citalopram dataset
review_classifier = ReviewClassifier('nb')
review_classifier.train('citalopram-reviews.csv')
review_classifier.evaluate_average_accuracy('citalopram-reviews.csv')
review_classifier.classify('neutral.txt')

# For Neural Network
review_classifier = NeuralNetReviewClassifier()
review_classifier.train('citalopram-reviews.csv')
review_classifier.classify('neutral.txt')
```

## Contributions

### Checklist

* Changes made/comitted/pushed in new branch
* Changes not far behind master
* Added comments and documentation to code
* Made sure styling matches Google style guide: <http://google.github.io/styleguide/pyguide.html>
* README updated if relevant changes made

### Making changes locally

1. Copy the URL from the Medinify repository and use Git to clone the repo:

    ```bash
    # Clone the repo into current directory
    git clone https://github.com/NanoNLP/medinify.git
    # Navigate to the newly cloned directory
    cd medinify
    ```

2. Create a branch off of develop to contain your changes:

    ```bash
    git checkout -b <new-branch-name>
    ```

3. After making changes to files or adding new files to the project, stage your changes

    ```bash
    git add <filename>
    ```

4. Next, we record the changes made and provide a message describing the changes made so others can understand

    ```bash
    git commit -m "Description of changes made"
    ```

5. After committitng, make sure everything looks good with:

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

6. Finally, push the changes to the new branch origin:

```bash
# If the branch doesn't exist on GitHub yet
git push --set-upstream origin test

# If the branch already exists
git push
```

### Making Pull Requests

After following the steps above, you can make a pull request directly on the Medinify GitHub. It should be a pull request to merge your new branch into develop.

Add a title, a description, and then press the “Create pull request” button. If you are closing an issue, put "closes #14", if you had issue 14.

Navigate to the reviewers tab and request a reviewer to review the PR.

### Authors

Bridget McInnes, Jorge Vargas, Gabby Gurdin, Nathan West, Mark Groves

## More Info

* [VCU Natural Language Processing Lab](https://nlp.cs.vcu.edu/)     ![alt text](https://nlp.cs.vcu.edu/images/vcu_head_logo "VCU")
* [Nanoinformatics Vertically Integrated Projects](https://rampages.us/nanoinformatics/)