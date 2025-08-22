import requests
import spacy
import logging
import os
from dotenv import load_dotenv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logging.warning("spaCy model 'en_core_web_sm' not found. Downloading...")
    from spacy.cli import download

    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


def fetch_news_headlines(company_name):
    """Fetches real-time news headlines using a News API."""
    if not NEWS_API_KEY:
        logging.error("NEWS_API_KEY not set in .env file.")
        return []

    logging.info(f"Fetching unstructured data for {company_name}...")
    try:
        url = f"https://newsapi.org/v2/everything?q={company_name}&apiKey={NEWS_API_KEY}&language=en&pageSize=10"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        headlines = [article['title'] for article in data.get('articles', [])]
        return headlines
    except Exception as e:
        logging.error(f"Error fetching news for {company_name}: {e}")
        return []


def process_headlines(headlines):
    """Extracts entities and sentiment from news headlines using VADER."""
    processed_events = []
    for title in headlines:
        doc = nlp(title)

        # --- VADER SENTIMENT ANALYSIS ---
        sentiment_scores = analyzer.polarity_scores(title)
        compound_score = sentiment_scores['compound']

        if compound_score >= 0.05:
            sentiment = 'positive'
        elif compound_score <= -0.05:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'

        entities = [ent.text for ent in doc.ents if ent.label_ in ['ORG', 'GPE', 'PERSON']]
        processed_events.append({
            'headline': title,
            'sentiment': sentiment,
            'sentiment_score': compound_score,  # Include the numeric score
            'entities': entities,
            'event_type': 'financial_event'
        })
    return processed_events