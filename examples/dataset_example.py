"""
Examples for how to use the Medinify ReviewDataset() class
"""

from medinify.datasets import Dataset

def main():
    """ Main function.
    """

    review_dataset = ReviewDataset('Citalopram', 'WebMD')
    url = 'https://www.webmd.com/drugs/drugreview-1701-citalopram-oral.aspx?drugid=1701&drugname=citalopram-oral'
    review_dataset.collect(url)
    review_dataset.save()
    # review_dataset.load()
    # review_dataset.final_save()
    review_dataset.generate_rating_webmd()
    review_dataset.print_stats()
    review_dataset.print_meta()
    # review_dataset.collect_all_common_reviews()
    # review_dataset.write_file('csv')

if __name__ == "__main__":
    main()
