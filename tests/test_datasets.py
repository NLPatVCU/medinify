import os
import pytest
from medinify.datasets import ReviewDataset

@pytest.fixture()
def dataset():
    print("setup")
    doxil_dataset = ReviewDataset('DOXIL')

    yield doxil_dataset

    print("teardown")
    filenames = ['doxil-dataset.pickle', 'doxil-reviews.json', 'doxil-reviews.csv']

    for name in filenames:
        if os.path.exists(name):
            os.remove(name)

def test_init_name(dataset):
    assert dataset.drug_name == 'doxil'

def test_save(dataset):
    dataset.collect('https://www.webmd.com/drugs/drugreview-12120-Doxil-intravenous.aspx?drugid=12120&drugname=Doxil-intravenous', True)
    dataset.save()
    assert os.path.exists('doxil-dataset.pickle')
