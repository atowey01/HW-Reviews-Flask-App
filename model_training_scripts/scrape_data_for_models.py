# Initially I used this script to scrape review data from Hostelworld that I could
# use to train my sentiment models. However, I found that Hostelworld has a URL
# for all hostels that shows the json of the reviews. This script is not fully finished
# Both methods seems to miss a few reviews and does not get old reviews which is
# possibly due to my proxy being blocked. Possibly could also try use Selenium
# rather than request HTML and see if that makes a difference. Could look at this
# again to figure it out better but for now I will just take the reviews that
# I get from the json URLs.

from requests_html import HTMLSession
from bs4 import BeautifulSoup
from logger import logger
import pandas as pd
import time
from fp.fp import FreeProxy

# make sure the date is in the future
url = 'https://www.hostelworld.com/s?q=Rio%20De%20Janeiro,%20Brazil&country=Brazil&city=Rio%20De%20Janeiro&type=city&id=127&from=2023-07-13&to=2023-07-15&guests=2&page=1'
# Make the request to the website
session = HTMLSession()

response = session.get(url)
response.html.render(timeout=40, sleep=5) # need a sleep to let the render complete in time
assert len(response.html.absolute_links) > 55
# Parse the HTML content of the page
soup = BeautifulSoup(response.html.raw_html, 'html.parser')

hostel_links = []
hostels = soup.find_all('div', class_='property-card')
# can use {"class": "property-card"
# hostel_location = soup.find_all('label', class_='sr-only')[0].text
hostel_location = "test location"

for hostel in hostels:
    hostel_links.append(hostel.find('a').attrs['href'])

reviews_list = []

for link in hostel_links:
    review_link = 'https://www.hostelworld.com'+link+'&display=reviews'
    proxy = FreeProxy(country_id=['IE']).get()
    session.proxies = proxy
    hostel_response = session.get(review_link)
    hostel_response.html.render(timeout=40, sleep=5) # check do we need to do this

    hostel_soup = BeautifulSoup(hostel_response.html.html, 'html.parser')
    hostel_name = hostel_soup.find('div', class_='title-2').text.strip()
    logger.info(f'Scraping hostel: {hostel_name}')
    hostel_score = hostel_soup.find('div', class_='score orange big').text.strip()

    # try to click next page to get the rest of the reviews
    # Load the jQuery library
    java_script = """
        () => {
            const script = document.createElement('script');
            script.src = 'https://code.jquery.com/jquery-3.6.0.min.js';
            document.head.appendChild(script);
        }
    """
    # TODO can we do this at the start somewhere
    hostel_response.html.render(script=java_script, timeout=40, sleep=5)

    next_button = True
    review_page_count = 1
    # TODO seems to not change at page 6 - something to do with my ip being restriced, added proxy
    while next_button:
        hostel_soup = BeautifulSoup(hostel_response.html.html, 'html.parser')
        next_button = hostel_soup.find('div', {'class': 'pagination-item pagination-next'})
        reviews = hostel_soup.find_all('div', class_='review review-item')
        logger.info(f'Scraping reviews on page: {review_page_count}')
        for review in reviews:
            try:
                review_text = review.find('div', class_='review-notes body-2').text.strip()
                review_rating = review.find('div', class_='score orange medium').text.strip()
                review_date = review.find('div', class_='date body-3').text.strip()
                reviewer_data = review.find_all('li', class_="details body-3")[0].text
            except:
                # todo FIX THIS ERROR
                next
                logger.info(f'Issue processing review')

            # get all reviews from each page into the reviews_list
            reviews_list.append({"hostel_location": hostel_location,
                            "hostel_name": hostel_name,
                            "overall_rating": hostel_score,
                            "review": review_text,
                            "review_rating": review_rating,
                            "reviewer_info": reviewer_data,
                            "date": review_date
                            })

        tries = 0
        while tries < 3:
            try:
                # todo get this script working
                script = f"""const item = document.getElementsByClassName("pagination-item pagination-next");
                if (item) {
                  {'item[0].click();' * review_page_count}
                }
                """
                script = script.replace("'", "")
                hostel_response.html.render(script=script, timeout=40, sleep=5)
                # If the code above runs without raising an exception, we can break out of the loop
                break
            except:
                # Increment the number of tries
                tries += 1
                # You can add a delay before retrying if you want
                time.sleep(5)
        # hostel_response.html.render(script=script, timeout=40, sleep=5)
        review_page_count += 1

reviews_df = pd.DataFrame(reviews_list)
reviews_df.to_csv('reviews_df.csv')

