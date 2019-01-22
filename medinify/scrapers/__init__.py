"""Configure medinify.scrapers
"""
from medinify.scrapers.webmd_scraper import WebMDScraper
from medinify.scrapers.drugs_scraper import DrugsScraper

__all__ = (
    'WebMDScraper',
    'DrugsScraper',
)
