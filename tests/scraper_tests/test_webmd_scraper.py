
import pytest
from medinify.scrapers import WebMDScraper


def test_default_initialization():
    scraper = WebMDScraper()
    assert len(scraper.reviews) == 0
    assert not scraper.collect_urls
    assert not scraper.collect_user_ids


def test_initialization_parameters():
    scraper = WebMDScraper(collect_urls=True, collect_user_ids=True)
    assert scraper.collect_urls
    assert scraper.collect_user_ids


def test_incorrect_url():
    scraper = WebMDScraper()
    with pytest.raises(AssertionError):
        scraper.scrape_page('https://www.drugs.com/comments/aripiprazole/abilify.html?page=1')


def test_no_reviews():
    scraper = WebMDScraper()
    returned = scraper.scrape_page(
        'https://www.webmd.com/drugs/drugreview-155251-T-Plus-topical.aspx?drugid=155251&drugname=T-Plus-topical')
    assert returned == 0


def test_scrape_page_default_parameters():
    scraper = WebMDScraper()
    scraper.scrape_page('https://www.webmd.com/drugs/drugreview-64439-abilify.aspx?drugid=64439&drugname=abilify')
    data_collected = list(scraper.reviews[0].keys())
    assert len(data_collected) == 4
    assert 'comment' in data_collected
    assert 'rating' in data_collected
    assert 'date' in data_collected
    assert 'drug' in data_collected
    assert len(scraper.reviews) == 5


def test_scrape_page_with_parameters():
    scraper = WebMDScraper(collect_user_ids=True, collect_urls=True)
    scraper.scrape_page('https://www.webmd.com/drugs/drugreview-64439-abilify.aspx?drugid=64439&drugname=abilify')
    data_collected = list(scraper.reviews[0].keys())
    assert len(data_collected) == 6
    assert 'user id' in data_collected
    assert 'url' in data_collected
    assert len(scraper.reviews) == 5


def test_scrape_empty_reviews():
    scraper = WebMDScraper()
    scraper.scrape('https://www.webmd.com/drugs/drugreview-5659-'
                   'methotrexate-sodium-injection.aspx?drugid=5659&drugname=methotrexate-sodium-injection')
    num_reviews = len(scraper.reviews)
    scraper.scrape('https://www.webmd.com/drugs/drugreview-5659-'
                   'methotrexate-sodium-injection.aspx?drugid=5659&drugname=methotrexate-sodium-injection')
    assert num_reviews == len(scraper.reviews)


def test_scrape_invalid_url():
    scraper = WebMDScraper()
    returned = scraper.scrape('https://www.webmd.com/drugs/drugreview-bhilknhj')
    assert returned == 0


def test_scrape_default_parameter():
    scraper = WebMDScraper()
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
    scraper = WebMDScraper(collect_urls=True, collect_user_ids=True)
    scraper.scrape('https://www.webmd.com/drugs/drugreview-5659-'
                   'methotrexate-sodium-injection.aspx?drugid=5659&drugname=methotrexate-sodium-injection')
    assert len(scraper.reviews) > 5
    data_collected = list(scraper.reviews[0].keys())
    assert len(data_collected) == 6
    assert 'user id' in data_collected
    assert 'url' in data_collected


def test_get_url_real_drug_name():
    scraper = WebMDScraper()
    url = scraper.get_url('actos')
    assert url == 'https://www.webmd.com/drugs/drugreview-17410-Actos-oral.aspx?drugid=17410&drugname=Actos-oral'


def test_url_fake_drug_name():
    scraper = WebMDScraper()
    url = scraper.get_url('garbage')
    assert not url


def test_drug_name_with_space():
    scraper = WebMDScraper()
    url = scraper.get_url('Methotrexate Vial')
    assert url == 'https://www.webmd.com/drugs/drugreview-5659-' \
                  'methotrexate-sodium-injection.aspx?drugid=5659&drugname=methotrexate-sodium-injection'


def test_short_drug_name():
    scraper = WebMDScraper()
    url = scraper.get_url('ACE')
    assert not url


def test_name_with_numbers():
    scraper = WebMDScraper()
    url = scraper.get_url('12.5CPD-1DCPM-30PSE')
    assert url == 'https://www.webmd.com/drugs/drugreview-150612-dexchlorphen-p-phed-' \
                  'chlophedianol-oral.aspx?drugid=150612&drugname=dexchlorphen-p-phed-chlophedianol-oral'


def test_name_with_numbers_and_spaces():
    scraper = WebMDScraper()
    url = scraper.get_url('7-Keto DHEA powder')
    assert url == 'https://www.webmd.com/drugs/drugreview-149048-7-Keto-DHEA.aspx?drugid=149048&drugname=7-Keto-DHEA'
