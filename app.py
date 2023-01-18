
from flask import (Flask, render_template, url_for, flash,
                   redirect, request, abort, Blueprint, session)
import flask
from logger import logger
from bs4 import BeautifulSoup
import numpy as np
from logger import logger
import requests
import itertools
import re
import tensorflow as tf
from sentence_splitter import SentenceSplitter
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import numpy as np
from joblib import load


###################################################

app = Flask(__name__)

logger.info(f"Loading model files")
model = TFAutoModelForSequenceClassification.from_pretrained("hostelworld_sentiment_model")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
regression_model = load('rating_prediction_model.joblib')
splitter = SentenceSplitter(language='en')
logger.info(f"Model loaded")

@app.route('/')
def index():
    return flask.render_template('index.html')

@app.route('/analyse-reviews')
def analyse_reviews_button():
    return render_template('analyse_reviews.html')

@app.route('/reviews-analysed')
def reviews_analysed():
    return render_template('reviews_analysed.html')


@app.route('/predict', methods=['POST'])
def predict():
    hostel_url = request.form['review_text']
    hostel_id = re.findall('(/)([0-9]+)(/|/?)', hostel_url )[0][1]
    hostel_url = f'https://www.hostelworld.com/hosteldetails/{hostel_id}'

    hostel_response = requests.get(hostel_url)
    hostel_soup = BeautifulSoup(hostel_response.text, 'html.parser')
    hostel_name = hostel_soup.find('title').text.strip().split(',')[0]
    hostel_overall_rating = hostel_soup.find('div', class_='score orange big').text.strip() if hostel_soup.find('div',
                                                                                                                class_='score orange big') is not None else "No Rating"
    image_url = hostel_soup.find_all(name='meta', property="og:image")[0]
    image_url = image_url.get("content")
    number_of_pages = 2
    reviews_list = []
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
                    "review_text": review['notes'],
                    "review_rating": review['summary']['overall'],
                    "stay_date": review['summary']['stayed'],
                    "review_text_split": [x for x in splitter.split(text=review['notes']) if x !='\r'],
                    "sentiments": [],
                    "predicted_review_rating": None,
                    "actual_to_predicted_rating_difference": None
                })
    logger.info('Getting Sentiment Predictions')
    for review in reviews_list:
        for sentence in review['review_text_split']:
            inputs = tokenizer(sentence, return_tensors="tf")
            logits = model(**inputs).logits
            predicted_class_id = int(tf.math.argmax(logits, axis=-1)[0])
            predicted_sentiment = model.config.id2label[predicted_class_id]
            predicted_sentiment_probability = float(tf.reduce_max(tf.nn.softmax(logits), axis=-1))
            review['sentiments'].append({"sentence": sentence,
                                         "predicted_sentiment": predicted_sentiment,
                                         "predicted_sentiment_probability": predicted_sentiment_probability})

    logger.info('Predicting overall review rating')
    for review in reviews_list:
        count_positive_sentences = 0
        count_negative_sentences = 0
        count_other_sentences = 0
        for sentence in review['sentiments']:
            if sentence['predicted_sentiment'] == "POSITIVE":
                count_positive_sentences += 1
            elif sentence['predicted_sentiment'] == "NEGATIVE":
                count_negative_sentences += 1
            else:
                count_other_sentences += 1

            input_data = np.array([
                count_positive_sentences,
                count_negative_sentences,
                count_other_sentences]).reshape(1, -1)

        predicted_review_rating = regression_model.predict(input_data) * 10
        review['predicted_review_rating'] = round(float(predicted_review_rating), 1)
        review['actual_to_predicted_rating_difference'] = round(
            float(predicted_review_rating - float(review['review_rating'])), 1)

    all_review_ratings = [review['predicted_review_rating'] for review in reviews_list]
    median_review_rating = np.median(all_review_ratings)

    sentiments = [review['sentiments'] for review in reviews_list]
    all_sentence_sentiments = list(itertools.chain.from_iterable(sentiments))
    positive_reviews = [x['sentence'] for x in all_sentence_sentiments if x['predicted_sentiment'] == 'POSITIVE']
    negative_reviews = [x['sentence'] for x in all_sentence_sentiments if x['predicted_sentiment'] == 'NEGATIVE']

    biggest_difference = sorted([x['actual_to_predicted_rating_difference'] for x in reviews_list])[:3]
    possible_different_reviews = [x for x in reviews_list if
                                  x['actual_to_predicted_rating_difference'] in biggest_difference]
    possible_different_reviews = sorted(possible_different_reviews, key=lambda key_name: key_name['actual_to_predicted_rating_difference'])

    return flask.render_template('reviews_analysed.html',
                                 hostel_name=hostel_name,
                                 hostel_overall_rating=hostel_overall_rating,
                                 predicted_rating=median_review_rating,
                                 positive_reviews=positive_reviews,
                                 negative_reviews=negative_reviews,
                                 possible_different_reviews=possible_different_reviews,
                                 image_url=image_url)


if __name__ == '__main__':
    app.run()
    # TODO remove debug when working
    # app.run(debug=True, port=8081)
    # app.run(host='localhost', port=8081)
