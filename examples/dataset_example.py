"""
Examples for how to use the Medinify dataset functionality
"""

from medinify.datasets import Dataset, SentimentDataset


def main():
    # To collect and write a data file
    dataset1 = SentimentDataset()  # use can specify scraper type and/or whether or not to collect urls and user ids
    dataset1.collect('https://www.webmd.com/drugs/drugreview-9027-Biaxin-oral.aspx?drugid=9027&drugname=Biaxin-oral')
    # dataset.write_file('./data/biaxin.csv')

    # To load an extant data file
    dataset2 = Dataset()  # you can specify the name of the csv column containing text data and label data
    dataset2.load_file('<some_csv>')  # file name must exist on your system
    dataset2.print_stats()


if __name__ == '__main__':
    main()
