"""
Examples for how to use the Medinify ReviewDataset() class
"""

from medinify.datasets import ReviewDataset

def main():
    """ Main function.
    """

    review_dataset = ReviewDataset('Citalopram')
    review_dataset.collect('https://www.webmd.com/drugs/drugreview-1701-citalopram-oral.aspx?drugid=1701&drugname=citalopram-oral', True)
    review_dataset.save()
    # review_dataset.load()
    # review_dataset.cleanse()
    review_dataset.print()
    # review_dataset.write_file('JSON')
    # review_dataset.write_file('csv')

if __name__ == "__main__":
    main()
