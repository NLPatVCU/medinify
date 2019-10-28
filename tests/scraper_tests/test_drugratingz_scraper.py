
# TODO (Dunya) - Add tests for DrugRatingz Scraper

import pytest
from medinify.scrapers import DrugRatingzScraper


def test_default_initialization():
    scraper = DrugRatingzScraper()
    assert len(scraper.reviews) == 0
    assert not scraper.collect_urls


def test_initialization_parameters():
    scraper = DrugRatingzScraper(collect_urls=True)
    assert scraper.collect_urls


def test_incorrect_url():
    scraper = DrugRatingzScraper()
    with pytest.raises(AssertionError):
        scraper.scrape_page('https://www.drugs.com/comments/aripiprazole/abilify.html?page=1')


def test_no_reviews():
    scraper = DrugRatingzScraper()
    returned = scraper.scrape_page('https://www.drugratingz.com/reviews/18088/Drug-Affinitor.html')
    assert returned == None


def test_scrape_correct_review_data():
    scraper = DrugRatingzScraper(collect_urls=True)
    scraper.scrape('https://www.drugratingz.com/reviews/141/Drug-Prednisone.html')
    assert scraper.reviews[-1]['comment'][:10] == 'This is a '
    assert scraper.reviews[-1]['comment'][-10:] == 'ing on it.'
    assert scraper.reviews[-1]['rating']['effectiveness'] == 5
    assert scraper.reviews[-1]['rating']['no side effects'] == 1
    assert scraper.reviews[-1]['rating']['convenience'] == 1
    assert scraper.reviews[-1]['rating']['value'] == 3
    assert scraper.reviews[-1]['date'] == '8/18/05'


def test_scrape_page_default_parameters():
    scraper = DrugRatingzScraper()
    scraper.scrape_page('https://www.drugratingz.com/reviews/141/Drug-Prednisone.html')
    data_collected = list(scraper.reviews[0].keys())
    assert len(data_collected) == 4
    assert 'comment' in data_collected
    assert 'rating' in data_collected
    assert 'date' in data_collected
    assert 'drug' in data_collected
    assert len(scraper.reviews) == 13


def test_scrape_page_with_parameters():
    scraper = DrugRatingzScraper(collect_urls=True)
    scraper.scrape_page('https://www.drugratingz.com/reviews/141/Drug-Prednisone.html')
    data_collected = list(scraper.reviews[0].keys())
    assert len(data_collected) == 5
    assert 'url' in data_collected


def test_scrape_empty_reviews():
    scraper = DrugRatingzScraper()
    scraper.scrape('https://www.drugratingz.com/reviews/258/Drug-Glucophage.html')
    num_reviews = len(scraper.reviews)
    scraper.scrape('https://www.drugratingz.com/reviews/258/Drug-Glucophage.html')
    assert num_reviews == len(scraper.reviews)


def test_scrape_invalid_url_no_title():
    scraper = DrugRatingzScraper()
    returned = scraper.scrape('https://www.drugratingz.com/reviews/258/Drug.html')
    assert returned == None


def test_scrape_default_parameter():
    scraper = DrugRatingzScraper()
    scraper.scrape('https://www.drugratingz.com/reviews/472/Drug-Lutera.html')
    assert len(scraper.reviews) > 10
    data_collected = list(scraper.reviews[0].keys())
    assert len(data_collected) == 4
    assert 'comment' in data_collected
    assert 'rating' in data_collected
    assert 'date' in data_collected
    assert 'drug' in data_collected


def test_scrape_with_parameters():
    scraper = DrugRatingzScraper(collect_urls=True)
    scraper.scrape('https://www.drugratingz.com/reviews/472/Drug-Lutera.html')
    assert len(scraper.reviews) > 10
    data_collected = list(scraper.reviews[0].keys())
    assert len(data_collected) == 5
    assert 'url' in data_collected


def test_scrape_assert_title_error():
    scraper = DrugRatingzScraper()
    returned = scraper.scrape('https://www.drugratingz.com/ShowThingCats.jsp')
    assert returned == 0


def test_scrape_no_reviews():
    scraper = DrugRatingzScraper()
    scraper.scrape('https://www.drugratingz.com/reviews/18589/Drug-Adderall.html')
    assert len(scraper.reviews) == 0


def test_get_rul_real_drug_name():
    scraper = DrugRatingzScraper()
    url = scraper.get_url('actos')
    assert url == 'https://www.drugratingz.com/reviews/340/Drug-Actos.html'


def test_url_fake_drug_name():
    scraper = DrugRatingzScraper()
    url = scraper.get_url('garbage')
    assert not url


def test_drug_name_with_space():
    scraper = DrugRatingzScraper()
    url = scraper.get_url('Ortho Tri-Cyclen Lo ')
    assert url == 'https://www.drugratingz.com/reviews/163/Drug-Ortho-Tri-Cyclen-Lo.html'



def test_short_drug_name():
    scraper = DrugRatingzScraper()
    url = scraper.get_url('ACE')
    assert not url


def test_name_with_numbers():
    scraper = DrugRatingzScraper()
    url = scraper.get_url('FPE 1070 (IS) Fpf 1070 (IS)')
    assert url == 'https://www.drugratingz.com/reviews/18813/Drug-Renacenz.html'


def test_name_with_numbers_and_spaces():
    scraper = DrugRatingzScraper()
    url = scraper.get_url('Ortho Cyclen 28')
    assert url == 'https://www.drugratingz.com/reviews/229/Drug-Ortho-Cyclen-28.html'

