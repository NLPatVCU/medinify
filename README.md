# Medinify

![Test Image 1](readmeassets/nlplab.png)

Medical text classification. 

## Requirements

* Python 3.6

## Getting Started
### Using Git Version Control to grab Medinify
[Ensure that you have git installed.](https://git-scm.com/downloads)

In Mac or Linux terminal:
```
git clone https://github.com/NLPatVCU/medinify.git
```
### Virtual environment setup
In order to manage dependencies/configuration [virtual environments](https://docs.python.org/3/tutorial/venv.html) should be used.  Ensure your current directory is the project installation directory. Then enter:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

## Workflow

Medinify is a general tool for medical text classification that also includes functionality for collecting drug review sentiment datasets from multiple online drug forums.

All datasets used with Medinify must be in .csv format, and have a text column (containing the text being labelled) and a label column (containing the text labels). 

If the data in the .csv file requires some extra processesing (i.e., if the labels are non-numeric, or if certain texts need to be removed) that functionality can be added by subclassing the Dataset class. SentimentDataset is an example of this, which is used for transforming star rating labels into sentiment labels.

### Collecting

If you want to use Medinify's scraping functionality to collect drug review sentiment datasets:

Datasets can be collected from three sources: a url, a .txt file containing a list of urls, or a .txt file containing a list of drug names.
The method for collection for each is shown below:

```python
from medinify.datasets import SentimentDataset
dataset = SentimentDataset()  
"""
The default scraper is 'webmd', but the 'scraper' argument could also be set to 'everydayhealth', 'drugs', or 'drugratingz'
There are also 'collect_user_id' and 'collect_urls' arguments, both default false
"""

# For collecting from url
dataset.collect('<valid_url>')

# For collecting from .txt url file
dataset.collect_from_urls(urls_file='path/to/urls/file')

# For collecting from .txt drug names file
dataset.collect_from_drug_names(drug_names_file='path/to/drug/names/file')

# To save .csv file
dataset.write_file('output_file_name.csv')
```

### Loading Data

In order to load .csv file into a dataset, the text column and label column must be specified

```python
from medinify.datasets import Dataset

dataset = Dataset('path/to/csv', text_column='<text column name>', label_column='<label column name>')
```

### Training, Evaluating, and Classifying

Medinify provides functionality for training, evaluating, and classifying with Naive Bayes, 
Random Forest, Support Vector Machine, and Convolutional Neural Network classificaton models.

```python
from medinify.datasets import SentimentDataset
from medinify.classifiers import Classifier

# create a Dataset object and load data from .csv file
dataset = SentimentDataset('path/to/csv/file')

# create classifier
clf = Classifier()
"""
For classifiers, both 'learner' and 'representation' arguments can be specified
(they are 'nb' (NaiveBayes) and 'bow' (Bag-of-Word) by default)
All learners have a default representation that produces the best results. 
Another representation ('embeddings', 'bow', or 'matrix') can be specified, but be careful
because it may be incompatible with the learner
"""

# fit model
model = clf.fit(dataset)

# evaluate model
eval_dataset = SentimentDataset('path/to/eval/dataset')
clf.evaluate(eval_dataset, trained_model=model)

# classify using model
classification_dataset = SentimentDataset('path/to/dataset')
clf.classify(classification_dataset, output_file='output_file.txt', trained_model=model)
```

### Saving and Loading Models

Trained models can be saved and loaded as pickle files

```python
from medinify.datasets import SentimentDataset
from medinify.classifiers import Classifier

dataset = SentimentDataset('path/to/csv/file')
clf = Classifier()
model = clf.fit(dataset)

# save model
clf.save(model, 'path/to/save/model')

# load saved model
model2 = clf.load('saved/model/path')
```

### Validation

Medinify has functionality for k-fold cross validation

```python
from medinify.datasets import SentimentDataset
from medinify.classifiers import Classifier

dataset = SentimentDataset('path/to/csv/file')
clf = Classifier()
clf.validate(dataset, k_folds=5)
```

## Contribution Checklist

* Changes made/comitted/pushed in new branch
* Changes not far behind develop
* Added comments and documentation to code
* Made sure styling matches Google style guide: <http://google.github.io/styleguide/pyguide.html>
* README updated if relevant changes made

## Making changes locally

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

## Making Pull Requests

After following the steps above, you can make a pull request directly on the Medinify GitHub. It should be a pull request to merge your new branch into develop.

Add a title, a description, and then press the “Create pull request” button. If you are closing an issue, put "closes #14", if you had issue 14.

Navigate to the reviewers tab and request a reviewer to review the PR.

## Authors

Bridget McInnes, Jorge Vargas, Gabby Gurdin, Nathan West, Ishaan Thakur, Mark Groves

## More Info

* [VCU Natural Language Processing Lab](https://nlp.cs.vcu.edu/)     ![alt text](https://nlp.cs.vcu.edu/images/vcu_head_logo "VCU")
* [Nanoinformatics Vertically Integrated Projects](https://rampages.us/nanoinformatics/)
