import flask
from flask import (Flask, render_template, request)
import re
import numpy as np
from helper_functions import (get_hostel_general_details,
                              get_review_data,
                              get_positive_and_negative_sentences,
                              get_reviews_with_rating_different_from_predicted_rating)

app = Flask(__name__)
app.secret_key = "abc"  # TODO change this and put in private file

@app.route('/')
def index():
    return flask.render_template('index.html')

@app.route('/submit-url')
def analyse_reviews_button():
    return render_template('submit_url.html')

@app.route('/reviews-analysed', methods=['POST'])
def analyse_reviews():
    # first we need to get the url for the data we want to scrape (different to input url)
    hostel_url = request.form['review_text']
    # if we can't find the hostel_id ask for a different URL
    try:
        hostel_id = re.findall('(/)([0-9]+)(/|/?)', hostel_url )[0][1]
    except:
        return render_template('submit_url.html', error='Please enter a valid URL')
    hostel_url = f'https://www.hostelworld.com/hosteldetails/{hostel_id}'

    number_of_pages = 2
    hostel_name, hostel_overall_rating, image_url = get_hostel_general_details(hostel_url)
    reviews_list = get_review_data(number_of_pages, hostel_id, hostel_name)
    all_review_ratings = [review['predicted_review_rating'] for review in reviews_list]
    median_review_rating = np.median(all_review_ratings)
    positive_reviews, negative_reviews = get_positive_and_negative_sentences(reviews_list)
    possible_different_reviews = get_reviews_with_rating_different_from_predicted_rating(reviews_list)

    return flask.render_template('reviews_analysed.html',
                                 hostel_name=hostel_name,
                                 hostel_overall_rating=hostel_overall_rating,
                                 predicted_rating=median_review_rating,
                                 positive_reviews=positive_reviews,
                                 negative_reviews=negative_reviews,
                                 possible_different_reviews=possible_different_reviews,
                                 image_url=image_url)