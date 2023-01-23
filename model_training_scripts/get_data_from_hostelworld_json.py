# Hostelworld provides a URL for each hostel showing the reviews in JSON form. For some
# reason it does not show all reviews for every hostel so for now I am using the reviews
# that are retrieved from the json. I have another script that manually scrapes the reviews
# but this also did not return all reviews. Could look into both script further to try get
# all reviews.

from bs4 import BeautifulSoup
from logger import logger
import pandas as pd
import requests
import re

reviews_list = []
number_of_pages = 7 # number of pages to get reviews from, seem to only be able to do 6 pages possibly due to proxies
city_country_dict = {"Bogota": "Colombia",
                     "Lima": "Peru",
                     "LaPaz": "Bolivia",
                     "Santiago": "Chile",
                     "Montevideo": "Uruguay",
                     "Buenos Aires": "Argentina",
                     "Rio De Janeiro": "Brazil",
                     "Bangkok": "Thailand",
                     "Phonm Penh": "Cambodia",
                     "Ho Chi Minh": "Vietnam",
                     "Dublin": "Ireland",
                     "Paris": "France",
                     "Rome": "Italy",
                     "Sydney": "Australia",
                     "Beijing": "China",
                     "Tokyo": "Japan",
                     "New York": "USA",
                     "Nairobi": "Kenya",
                     "Mumbai": "India",
                     "Addis Ababa": "Ethiopia"}

# we will iterate through all hostels in a couple of cities and get the reviews for each hostel
for city, country in city_country_dict.items():
    logger.info(f'Processing hostels in {city}, {country}')
    # getting the html info to be used
    url = f'https://www.hostelworld.com/hostels/{city}'
    response = requests.get(url)

    # create soup
    soup = BeautifulSoup(response.text, 'html.parser')
    hostel_links = []
    hostels = soup.find_all('div', class_='property-card')
    hostel_links = [hostel.find('a').attrs['href'] for hostel in hostels]
    hostel_ids = [re.findall('(p/)([0-9]+)(/)', link)[0][1] for link in hostel_links]

    for hostel_id in hostel_ids:
        hostel_url = f'https://www.hostelworld.com/hosteldetails/{hostel_id}'
        hostel_response = requests.get(hostel_url)
        hostel_soup = BeautifulSoup(hostel_response.text, 'html.parser')
        hostel_name = hostel_soup.find('title').text.strip().split(',')[0]
        hostel_overall_rating = hostel_soup.find('div', class_='score orange big').text.strip() if hostel_soup.find('div', class_='score orange big') is not None else "No Rating"
        for page_number in range(1, number_of_pages):
            logger.info(f"Processing reviews for hostel {hostel_name} on page {page_number}")
            hostel_review_url = f'https://www.hostelworld.com/properties/{hostel_id}/reviews?sort=newest&page={page_number}'
            requests_json = requests.get(hostel_review_url).json()
            if 'reviews' in requests_json:
                hostel_reviews = requests.get(hostel_review_url).json()['reviews']
            else:
                # if there are no review we can break out of check the next pages
                break
            if hostel_reviews is not None:
                for review in hostel_reviews:
                    reviews_list.append({
                        "review_id": review['id'],
                        "city": city,
                        "country": country,
                        "hostel_id": hostel_id,
                        "hostel_url": hostel_url,
                        "hostel_name": hostel_name,
                        "hostel_overall_rating": hostel_overall_rating,
                        "review_text": review['notes'],
                        "review_rating": review['summary']['overall'],
                        "reviewer_nationality": review['reviewer']['nationality'],
                        "reviewer_gender": review['reviewer']['gender'],
                        "reviewer_age_group": review['reviewer']['ageGroup'],
                        "stay_date": review['summary']['stayed']
                    })

reviews_df = pd.DataFrame(reviews_list)
logger.info("Saving Excel File")
reviews_df.to_excel('hostel_reviews_df.xlsx', engine='openpyxl', index=False)