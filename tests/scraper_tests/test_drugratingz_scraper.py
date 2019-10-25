
# TODO (Dunya) - Add tests for DrugRatingz Scraper

import pytest
from medinify.scrapers import DrugRatingzScraper


def test_default_initialization():
    scraper = DrugRatingzScraper()
    assert len(scraper.reviews) == 0
    assert not scraper.collect_urls

