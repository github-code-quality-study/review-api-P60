import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from urllib.parse import parse_qs, urlparse
import json
import pandas as pd
from datetime import datetime
import uuid
import os
from typing import Callable, Any
from wsgiref.simple_server import make_server

nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)

# Initialize sentiment analyzer and stopwords
adj_noun_pairs_count = {}
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

# Load reviews from CSV file
reviews = pd.read_csv('data/reviews.csv').to_dict('records')

# Define allowed locations for filtering
allowed_locations = [
    "Albuquerque, New Mexico",
    "Carlsbad, California",
    "Chula Vista, California",
    "Colorado Springs, Colorado",
    "Denver, Colorado",
    "El Cajon, California",
    "El Paso, Texas",
    "Escondido, California",
    "Fresno, California",
    "La Mesa, California",
    "Las Vegas, Nevada",
    "Los Angeles, California",
    "Oceanside, California",
    "Phoenix, Arizona",
    "Sacramento, California",
    "Salt Lake City, Utah",
    "San Diego, California",
    "Tucson, Arizona"
]

class ReviewAnalyzerServer:
    def __init__(self) -> None:
        # This method is a placeholder for future initialization logic
        pass

    def analyze_sentiment(self, review_body):
        sentiment_scores = sia.polarity_scores(review_body)
        return sentiment_scores

    def __call__(self, environ: dict[str, Any], start_response: Callable[..., Any]) -> bytes:
        """
        The environ parameter is a dictionary containing some useful
        HTTP request information such as: REQUEST_METHOD, CONTENT_LENGTH, QUERY_STRING,
        PATH_INFO, CONTENT_TYPE, etc.
        """

        if environ["REQUEST_METHOD"] == "GET":
            # Parse query string parameters
            query_string = environ.get("QUERY_STRING", "")
            query_params = parse_qs(query_string)
            
            location = query_params.get("location", [None])[0]
            start_date = query_params.get("start_date", [None])[0]
            end_date = query_params.get("end_date", [None])[0]

            filtered_reviews = reviews

            # Filter by location if specified and valid
            if location:
                if location not in allowed_locations:
                    start_response("400 Bad Request", [("Content-Type", "application/json")])
                    return [json.dumps({"error": "Invalid location"}).encode("utf-8")]
                filtered_reviews = [review for review in filtered_reviews if review['Location'] == location]

            # Filter by start date if specified
            if start_date:
                start_date = datetime.strptime(start_date, '%Y-%m-%d')
                filtered_reviews = [review for review in filtered_reviews if datetime.strptime(review['Timestamp'], '%Y-%m-%d %H:%M:%S') >= start_date]

            # Filter by end date if specified
            if end_date:
                end_date = datetime.strptime(end_date, '%Y-%m-%d')
                filtered_reviews = [review for review in filtered_reviews if datetime.strptime(review['Timestamp'], '%Y-%m-%d %H:%M:%S') <= end_date]

            # Analyze sentiment for each filtered review
            for review in filtered_reviews:
                review['sentiment'] = self.analyze_sentiment(review['ReviewBody'])
            
            # Sort reviews by sentiment compound score in descending order
            filtered_reviews.sort(key=lambda x: x['sentiment']['compound'], reverse=True)
            response_body = json.dumps(filtered_reviews, indent=2).encode("utf-8")

            # Set the appropriate response headers
            start_response("200 OK", [("Content-Type", "application/json"), ("Content-Length", str(len(response_body)))])
            return [response_body]


        if environ["REQUEST_METHOD"] == "POST":
            try:
                # Read and parse the POST request body
                content_length = int(environ.get('CONTENT_LENGTH', '0'))
                request_body = environ['wsgi.input'].read(content_length)
                post_params = parse_qs(request_body.decode('utf-8'))

                review_body = post_params.get("ReviewBody", [None])[0]
                location = post_params.get("Location", [None])[0]

                # Validate required parameters
                if not review_body or not location:
                    start_response("400 Bad Request", [("Content-Type", "application/json")])
                    return [json.dumps({"error": "ReviewBody and Location are required"}).encode("utf-8")]
                
                # Validate location
                if location not in allowed_locations:
                    start_response("400 Bad Request", [("Content-Type", "application/json")])
                    return [json.dumps({"error": "Invalid location"}).encode("utf-8")]

                # Create a new review entry
                review_id = str(uuid.uuid4())
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                new_review = {
                    "ReviewId": review_id,
                    "ReviewBody": review_body,
                    "Location": location,
                    "Timestamp": timestamp
                }

                # Append new review to the list
                reviews.append(new_review)

                # Set the response for successful creation
                start_response("201 Created", [("Content-Type", "application/json")])
                return [json.dumps(new_review).encode("utf-8")]

            except Exception as e:
                # Handle any server errors
                start_response("500 Internal Server Error", [("Content-Type", "application/json")])
                return [json.dumps({"error": str(e)}).encode("utf-8")]

if __name__ == "__main__":
    app = ReviewAnalyzerServer()
    port = os.environ.get('PORT', 8000)
    with make_server("", port, app) as httpd:
        print(f"Listening on port {port}...")
        httpd.serve_forever()