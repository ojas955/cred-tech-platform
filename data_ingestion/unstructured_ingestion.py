import requests
import spacy
import logging

# Use a lighter model like `en_core_web_sm` for faster processing
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logging.warning("spaCy model 'en_core_web_sm' not found. Downloading...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# --- Mock Data for Demonstration ---
MOCK_NEWS_HEADLINES = {
    'AAPL': [
        "Apple announces record-breaking iPhone sales for Q4.",
        "Apple faces new antitrust lawsuit in Europe.",
        "Tim Cook hints at new AI features at WWDC.",
        "Analyst warns of slowing growth in China for Apple.",
    ],
    'MSFT': [
        "Microsoft acquires major gaming studio for $20B.",
        "Microsoft to lay off 1,000 employees in restructuring plan.",
        "New Azure cloud services announced at Ignite conference.",
    ],
    'GOOG': [
        "Google's parent company Alphabet posts strong quarterly earnings.",
        "Google invests $500M in new data centers.",
        "Regulatory scrutiny over Google's ad business intensifies.",
    ],
}

def fetch_news_headlines(company_name):
    """
    Mocks fetching real-time news headlines for a company.
    In a real-world scenario, you'd use a real news API here.
    """
    logging.info(f"Mocking fetch for unstructured data for {company_name}...")
    return MOCK_NEWS_HEADLINES.get(company_name, [])

def process_headlines(headlines):
    """Extracts entities and sentiment from news headlines."""
    processed_events = []
    for title in headlines:
        doc = nlp(title)

        # Simple rule-based event classification
        sentiment = 'neutral'
        if any(word in title.lower() for word in ['acquisition', 'invests', 'expansion', 'record-breaking', 'strong']):
            sentiment = 'positive'
        elif any(word in title.lower() for word in ['lawsuit', 'lay off', 'restructuring', 'warns', 'slowing', 'regulatory scrutiny']):
            sentiment = 'negative'

        # Extract company entities
        entities = [ent.text for ent in doc.ents if ent.label_ in ['ORG', 'GPE']]

        processed_events.append({
            'headline': title,
            'sentiment': sentiment,
            'entities': entities,
            'event_type': 'financial_event'
        })
    return processed_events
