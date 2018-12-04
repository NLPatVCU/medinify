from urllib.request import urlopen
import requests
from bs4 import BeautifulSoup
import re
import time
import csv
import argparse


if __name__ == "__main__":
    ## Parse input arguments
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
    my_list = []

    for i in range(num_pages):
        url = quote_page1 + str(i) + quote_page2
        headers = {'User-Agent': 'Mozilla/5.0'}
        page = requests.get(url)
        soup = BeautifulSoup(page.text, 'html.parser')
        reviews = soup.find_all('div', attrs={'class': 'userPost'})

        for r in reviews:
            com = r.find('p', id=re.compile("^comFull*"))
            if com is not None:
                com = com.text.replace('Comment:', '').replace('Hide Full Comment', '')
                rate = int(r.find_all('span', attrs={'class': 'current-rating'})[0].text.replace('Current Rating:', '').strip())
                rate = rate + int(r.find_all('span', attrs={'class': 'current-rating'})[1].text.replace('Current Rating:', '').strip())
                rate = rate + int(r.find_all('span', attrs={'class': 'current-rating'})[2].text.replace('Current Rating:', '').strip())
                rate = int(rate) / 3.0
                my_list.append({'comment': com, 'rating': rate})
        # time.sleep(1)
    print(len(my_list))

    print(my_list[0])

    keys = my_list[0].keys()
    with open(args.o, 'w') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(my_list)