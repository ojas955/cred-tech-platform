import requests
import spacy
# Use a lighter model like `en_core_web_sm` for faster processing
nlp = spacy.load("en_core_web_sm")

def fetch_news_headlines(company_name):
    """Fetches real-time news headlines for a company."""
    # This is a placeholder; you'd use a real news API here.
    # e.g., using NewsAPI or a similar free source.
    api_url = f"https://api.news.com/v1/articles?q={company_name}"
    response = requests.get(api_url)
    if response.status_code == 200:
        return [article['title'] for article in response.json()['articles']]
    return []

def process_headlines(headlines):
    """Extracts entities and sentiment from news headlines."""
    processed_events = []
    for title in headlines:
        doc = nlp(title)

        # Simple rule-based event classification
        if any(word in title.lower() for word in ['debt', 'default', 'restructuring']):
            sentiment = 'negative'
        elif any(word in title.lower() for word in ['acquisition', 'profit', 'expansion']):
            sentiment = 'positive'
        else:
            sentiment = 'neutral'

        # Extract company entities
        entities = [ent.text for ent in doc.ents if ent.label_ in ['ORG', 'GPE']]

        processed_events.append({
            'headline': title,
            'sentiment': sentiment,
            'entities': entities,
            'event_type': 'financial_event'
        })
    return processed_events