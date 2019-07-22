"""Configure medinify.scrapers
"""
from medinify.scrapers.webmd_scraper import WebMDScraper
from medinify.scrapers.drugs_scraper import DrugsScraper
from medinify.scrapers.drugratingz_scraper import DrugRatingzScraper
from medinify.scrapers.everydayhealth_scraper import EverydayHealthScraper
from medinify.scrapers.scraper import Scraper


__all__ = (
    'WebMDScraper',
    'DrugsScraper',
    'DrugRatingzScraper',
    'EverydayHealthScraper',
    'Scraper'
)
