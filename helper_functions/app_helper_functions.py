from bs4 import BeautifulSoup
from logger import logger
import requests
import tensorflow as tf
from sentence_splitter import SentenceSplitter
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import numpy as np
from joblib import load
import itertools

logger.info(f"Loading model files")
model = TFAutoModelForSequenceClassification.from_pretrained("atowey01/hostel-reviews-sentiment-model")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
regression_model = load('models/rating_prediction_model.joblib')
splitter = SentenceSplitter(language='en')
logger.info(f"Model loaded")


def get_hostel_general_details(hostel_url: str):
    """
    Function to get hostel details from our scraping
    Args: hostel_url(str): hostel url user inputs
    Returns: hostel name(str), hostel_overall_rating(str), image_url(str)

    """
    hostel_response = requests.get(hostel_url)
    hostel_soup = BeautifulSoup(hostel_response.text, 'html.parser')
    hostel_name = hostel_soup.find('title').text.strip().split(',')[0]
    hostel_overall_rating = hostel_soup.find('div', class_='score orange big').text.strip() if hostel_soup.find('div',                                                                                                     class_='score orange big') is not None else "No Rating"
    image_url = hostel_soup.find_all(name='meta', property="og:image")[0]
    image_url = image_url.get("content")

    return hostel_name, hostel_overall_rating, image_url


def get_review_sentiment_predictions(reviews_list):
    """
    Use model to predict the sentiment of the hostel reviews and add to the reviews_list
    Args: reviews_list(list)
    Returns: reviews_list(list)

    """
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

    return reviews_list


def get_review_data(number_of_pages:int, hostel_id:str, hostel_name: str) -> list:
    """
    Function to get hostel details from our scraping
    Args: number_of_pages(int): number of pages to get reviews from (10 on each)
          hostel_id(str): hostel_id we are analysing
          hostel_name(str)
    Returns: reviews_list(list): a list of dicts containing all relevant review data
    """
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
                    "review_text_split": [x for x in splitter.split(text=review['notes']) if x !='\r' and x != ''],
                    "sentiments": [],
                    "predicted_review_rating": None,
                    "actual_to_predicted_rating_difference": None
                })
    reviews_list = get_review_sentiment_predictions(reviews_list)
    reviews_list = get_review_rating_prediction(reviews_list)

    return reviews_list


def get_review_rating_prediction(reviews_list):
    """
    Function to predict the rating of a review based on the text in the review
    Args: review_list(list)
    Returns: reviews_list(list), predicted_review_rating(float)
    """

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

    return reviews_list


def get_positive_and_negative_sentences(reviews_list: list):
    """
    Function to get the positive and negative sentences in the reviews
    Args: review_list(list)
    Returns: positive_reviews(list[str]), predicted_review_rating(list[str])
    """
    sentiments = [review['sentiments'] for review in reviews_list]
    all_sentence_sentiments = list(itertools.chain.from_iterable(sentiments))
    positive_reviews = [x['sentence'] for x in all_sentence_sentiments if x['predicted_sentiment'] == 'POSITIVE']
    negative_reviews = [x['sentence'] for x in all_sentence_sentiments if x['predicted_sentiment'] == 'NEGATIVE']

    return positive_reviews, negative_reviews


def get_reviews_with_rating_different_from_predicted_rating(reviews_list: list) -> list:
    """
    Function to get the reviews where the actual rating may be higher than we would expect from the review text
    Args: review_list(list)
    Returns: possible_different_reviews(list)
    """
    biggest_difference = sorted([x['actual_to_predicted_rating_difference'] for x in reviews_list])[:3]
    possible_different_reviews = [x for x in reviews_list if
                                  x['actual_to_predicted_rating_difference'] in biggest_difference]
    possible_different_reviews = sorted(possible_different_reviews,
                                        key=lambda key_name: key_name['actual_to_predicted_rating_difference'])

    return possible_different_reviews
