
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
    returned = scraper.scrape_page(
        'https://www.webmd.com/drugs/drugreview-155251-T-Plus-topical.aspx?drugid=155251&drugname=T-Plus-topical')
    assert returned == 0


def test_scrape_correct_review_data():
    scraper = DrugRatingzScraper(collect_user_ids=True)
    scraper.scrape('url')
    assert scraper.reviews[-1]['comment'][:10] == 'I started '
    assert scraper.reviews[-1]['comment'][-10:] == 'vitamin :)'
    assert scraper.reviews[-1]['user id'] == 'A95, 13-18 Female  on Treatment for 1 to 6 months (Patient)'
    assert scraper.reviews[-1]['rating']['effectiveness'] == 5
    assert scraper.reviews[-1]['rating']['ease of use'] == 5
    assert scraper.reviews[-1]['rating']['satisfaction'] == 5
    assert scraper.reviews[-1]['date'] == '10/6/2010 10:10:35 PM'


def test_scrape_page_default_parameters():
    scraper = DrugRatingzScraper()
    scraper.scrape_page('url')
    data_collected = list(scraper.reviews[0].keys())
    assert len(data_collected) == 4
    assert 'comment' in data_collected
    assert 'rating' in data_collected
    assert 'date' in data_collected
    assert 'drug' in data_collected
    assert len(scraper.reviews) == 5


def test_scrape_page_with_parameters():
    scraper = DrugRatingzScraper(collect_urls=True)
    scraper.scrape_page('url')
    data_collected = list(scraper.reviews[0].keys())
    assert len(data_collected) == 6
    assert 'user id' in data_collected
    assert 'url' in data_collected


def test_scrape_empty_reviews():
    scraper = DrugRatingzScraper()
    scraper.scrape('https://www.webmd.com/drugs/drugreview-5659-'
                   'methotrexate-sodium-injection.aspx?drugid=5659&drugname=methotrexate-sodium-injection')
    num_reviews = len(scraper.reviews)
    scraper.scrape('https://www.webmd.com/drugs/drugreview-5659-'
                   'methotrexate-sodium-injection.aspx?drugid=5659&drugname=methotrexate-sodium-injection')
    assert num_reviews == len(scraper.reviews)


def test_scrape_invalid_url_no_title():
    scraper = DrugRatingzScraper()
    returned = scraper.scrape('https://www.webmd.com/drugs/drugreview-bhilknhj')
    assert returned == 0


def test_scrape_default_parameter():
    scraper = DrugRatingzScraper()
    scraper.scrape('https://www.webmd.com/drugs/drugreview-5659-'
                   'methotrexate-sodium-injection.aspx?drugid=5659&drugname=methotrexate-sodium-injection')
    assert len(scraper.reviews) > 5
    data_collected = list(scraper.reviews[0].keys())
    assert len(data_collected) == 4
    assert 'comment' in data_collected
    assert 'rating' in data_collected
    assert 'date' in data_collected
    assert 'drug' in data_collected


def test_scrape_with_parameters():
    scraper = DrugRatingzScraper(collect_urls=True)
    scraper.scrape('https://www.webmd.com/drugs/drugreview-5659-'
                   'methotrexate-sodium-injection.aspx?drugid=5659&drugname=methotrexate-sodium-injection')
    assert len(scraper.reviews) > 5
    data_collected = list(scraper.reviews[0].keys())
    assert len(data_collected) == 6
    assert 'user id' in data_collected
    assert 'url' in data_collected


def test_scrape_assert_title_error():
    scraper = DrugRatingzScraper()
    returned = scraper.scrape('https://www.webmd.com/drugs/2/index')
    assert returned == 0


def test_scrape_no_reviews():
    scraper = DrugRatingzScraper()
    scraper.scrape('https://www.webmd.com/drugs/drugreview-174349-8HR-Muscle-Aches-'
                   'Pain-oral.aspx?drugid=174349&drugname=8HR-Muscle-Aches-Pain-oral')
    assert len(scraper.reviews) == 0


def test_get_rul_real_drug_name():
    scraper = DrugRatingzScraper()
    url = scraper.get_url('drugname')
    assert url == 'url'


def test_url_fake_drug_name():
    scraper = DrugRatingzScraper()
    url = scraper.get_url('garbage')
    assert not url


def test_drug_name_with_space():
    scraper = DrugRatingzScraper()
    url = scraper.get_url('Methotrexate Vial')
    assert url == 'url'



def test_short_drug_name():
    scraper = DrugRatingzScraper()
    url = scraper.get_url('ACE')
    assert not url


def test_name_with_numbers():
    scraper = DrugRatingzScraper()
    url = scraper.get_url('12.5CPD-1DCPM-30PSE')
    assert url == 'https://www.webmd.com/drugs/drugreview-150612-dexchlorphen-p-phed-' \
                  'chlophedianol-oral.aspx?drugid=150612&drugname=dexchlorphen-p-phed-chlophedianol-oral'


def test_name_with_numbers_and_spaces():
    scraper = DrugRatingzScraper()
    url = scraper.get_url('7-Keto DHEA powder')
    assert url == 'https://www.webmd.com/drugs/drugreview-149048-7-Keto-DHEA.aspx?drugid=149048&drugname=7-Keto-DHEA'

