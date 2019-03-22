"""Configure medinify.scrapers
"""
from medinify.scrapers.webmd_scraper import WebMDScraper
from medinify.scrapers.drugs_scraper import DrugsScraper
from medinify.scrapers.drugratingz_scraper import DrugRatingzScraper
from medinify.scrapers.iodine_scraper import IodineScraper
from medinify.scrapers.everydayhealth_scraper import EverydayHealthScraper


__all__ = (
    'WebMDScraper',
    'DrugsScraper',
    'DrugRatingzScraper',
    'EverydayHealthScraper',
)
