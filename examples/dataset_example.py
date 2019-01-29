"""
Examples for how to use the Medinify ReviewDataset() class
"""

from medinify.datasets import ReviewDataset

def main():
    """ Main function.
    """

    review_dataset = ReviewDataset('Citalopram')
    review_dataset.collect('https://www.webmd.com/drugs/drugreview-1701-citalopram-oral.aspx?drugid=1701&drugname=citalopram-oral')
    review_dataset.save()
    review_dataset.load()
    review_dataset.print()

if __name__ == "__main__":
    main()