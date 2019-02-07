"""Drug review scraper for Medinify 
    Iodine.com
    Last modified: 2/5/2019"""

import csv 
import requests 
from bs4 import BeautifulSoup 

class IodineScraper():
    """objective of script is to scrape iodine.com for drug reviews """

    def scraper(self, drug_url, output_path):
        """
        Args: drug_url: iodine.com page 
        output_path: file path for output 
        pages (int): number of pages to scrape from arg """
        
        #initialize 
        review_list = [] 
        #types of ratings (best -> worst)
        worth_it = 0
        worked_well = 0
        big_hassle = 0 

        #iter through multi-pages  
        page = requests.get(drug_url)
        soup = BeautifulSoup(page.text, 'html.parser')
        reviews = soup.find_all('div', {'class': 'tri bg-white relative depth-1 p-3 col mo-m-t-2'})
        for review in reviews: 
            if(len(review) >= 2):
                find_span = review.find_all('span')[1]
                #print ('find span:')
                
                comment = review.find_all('span')[1].text
                #print (comment)

                if review.find('div', {'class': "purple"}):
                    worth_it +=1 
                elif review.find('div', {'class': "navy-blue"}):
                    worked_well +=1
                elif review.find('div', {'class': "red-3"}):
                    big_hassle +=1

                #list of individual reviews 
                review_list.append({'review': review
                                    'worth it': worth_it,
                                    'worked well': worked_well, 
                                    'big hassle': big_hassle})

        
        with open(output_path, 'w') as output_file: 
            dict_writer = csv.DictWriter(output_file, ['comment', 
                                                        'worth it', 
                                                        'worked well',
                                                        'big hassle'])
            dict_writer.writeheader() 
            dict_writer.writerows(review_list)
        
        print('Reviews scraped: ' + str(len(review_list)))