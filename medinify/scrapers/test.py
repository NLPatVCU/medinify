from webmd_scraper import WebMDScraper

def main():
    """ Main function.
    """

    input_url = "https://www.webmd.com/drugs/drugreview-1701-citalopram-oral.aspx?drugid=1701&drugname=citalopram-oral"

    scraper = WebMDScraper("citalopram_train.csv")
    scraper.scrape(input_url, 10)

if __name__ == "__main__":
    main()