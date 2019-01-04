"""Drug review scraper for Medinify.

This module scrapes comments from WebMD along with their rating. Based
on work by Amy Olex 11/13/17.
"""

import re
import csv
import argparse
import requests
from bs4 import BeautifulSoup

def clean_comment(comment):
    """Cleans comment for proper CSV usage.

    Args:
        comment: Comment to be cleaned.
    Returns:
        The cleaned comment.
    """
    comment = comment.replace('Comment:', '').replace('Hide Full Comment', '')
    comment = ' '.join(comment.splitlines())
    return comment

def main():
    """Main method.
    """
    parser = argparse.ArgumentParser(description='Scrape Drug reviews from webmd.com')
    parser.add_argument('-i', metavar='inputurl', type=str,
                        help='URL for the desired drug', required=True)
    parser.add_argument('-o', metavar='ouputfile', type=str, help='path to file to output',
                        required=True)
    parser.add_argument('-p', metavar='pages', type=str, help='how many pages to go through',
                        required=False, default=1)
    args = parser.parse_args()

    quote_page1 = args.i + '&pageIndex='
    quote_page2 = '&sortby=3&conditionFilter=-500'

    num_pages = int(args.p)
    review_list = []

    for i in range(num_pages):
        url = quote_page1 + str(i) + quote_page2
        page = requests.get(url)
        soup = BeautifulSoup(page.text, 'html.parser')
        reviews = soup.find_all('div', attrs={'class': 'userPost'})

        for review in reviews:
            comment = review.find('p', id=re.compile("^comFull*")).text
            comment = clean_comment(comment)
            if comment:
                ratings = review.find_all('span', attrs={'class': 'current-rating'})
                calculated_rating = 0.0

                for rating in ratings:
                    calculated_rating += int(rating.text.replace('Current Rating:', '').strip())

                calculated_rating = calculated_rating / 3.0
                review_list.append({'comment': comment, 'rating': calculated_rating})


    with open(args.o, 'w') as output_file:
        dict_writer = csv.DictWriter(output_file, ['comment', 'rating'])
        dict_writer.writeheader()
        dict_writer.writerows(review_list)

    for review in review_list:
        print(review)

    print('Reviews scraped: ' + str(len(review_list)))

if __name__ == "__main__":
    main()
