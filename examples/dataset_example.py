"""
Examples for how to use the Medinify ReviewDataset() class
"""

from medinify.datasets import ReviewDataset

def main():
    """ Main function.
    """

    review_dataset = ReviewDataset('Citalopram')
    # review_dataset.collect('https://www.webmd.com/drugs/drugreview-52891-Adriamycin-intravenous.aspx?drugid=52891&drugname=Adriamycin-intravenous')
    # review_dataset.save()
    review_dataset.load()
    # review_dataset.generate_rating()
    # review_dataset.balance()
    # review_dataset.reviews = review_dataset.reviews[:2000]
    # review_dataset.print()
    # review_dataset.collect_all_common_reviews()
    # review_dataset.write_file('JSON')
    review_dataset.write_file('csv')
    review_dataset.print_stats()

if __name__ == "__main__":
    main()
