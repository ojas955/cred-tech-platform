# data_ingestion/data_pipeline.py

import sys
import pandas as pd
from datetime import datetime
import json
import logging

# Assuming these modules exist in the same directory structure
from data_ingestion.structured_ingestion import fetch_yahoo_finance_data
from data_ingestion.unstructured_ingestion import fetch_news_headlines, process_headlines
from models.scoring_engine import CreditScoringModel
from models.explainability_layer import generate_explanation

# --- Configuration and Setup ---
# Set up basic logging to see what the pipeline is doing
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# In a real-world scenario, you would have a list of companies to process
COMPANIES_TO_TRACK = ['AAPL', 'MSFT', 'GOOG']

# --- Main Pipeline Orchestration Function ---
def run_pipeline(company_ticker: str):
    """
    Executes the end-to-end data pipeline for a single company.

    Args:
        company_ticker (str): The stock ticker for the company to analyze.
    """
    logging.info(f"--- Starting pipeline for {company_ticker} ---")

    # 1. Data Ingestion
    logging.info("Step 1: Ingesting data...")
    try:
        # Structured data ingestion
        # NOTE: Using a mock function call for demonstration.
        # In a real app, you would use the live `fetch_yahoo_finance_data`
        structured_data = fetch_yahoo_finance_data(company_ticker)
        if not structured_data:
            logging.error("Failed to ingest structured data. Exiting.")
            return None

        # Unstructured data ingestion & processing
        headlines = fetch_news_headlines(company_ticker)
        unstructured_events = process_headlines(headlines)

    except Exception as e:
        logging.error(f"Data ingestion failed: {e}")
        return None

    # 2. Feature Engineering & Model Preparation
    logging.info("Step 2: Preparing data for the model...")
    # Get the latest financial data point from the structured data
    latest_financials = structured_data['financials'][-1]

    # Calculate sentiment score from unstructured data
    sentiment_score = sum(1 for e in unstructured_events if e['sentiment'] == 'positive') - \
                      sum(1 for e in unstructured_events if e['sentiment'] == 'negative')

    # Create the final feature vector for the model
    input_features = pd.DataFrame([{
        'debt_to_equity': latest_financials['debtToEquity'],
        'current_ratio': latest_financials['currentRatio'],
        'operating_margin': latest_financials['operatingMargin'],
        'sentiment_score': sentiment_score
    }])

    # 3. Adaptive Scoring Engine
    logging.info("Step 3: Running the scoring engine...")
    try:
        # The model is trained inside the class for this example.
        # In production, you would load a pre-trained model.
        scoring_model = CreditScoringModel()

        # The model needs to be trained before it can predict
        # This is a mock training dataset to make the example runnable
        X_train = pd.DataFrame([
            {'debt_to_equity': 0.5, 'current_ratio': 1.8, 'operating_margin': 0.12, 'sentiment_score': 2},
            {'debt_to_equity': 0.7, 'current_ratio': 1.5, 'operating_margin': 0.08, 'sentiment_score': -1},
        ])
        y_train = [1, 0] # 1 for good, 0 for bad credit
        scoring_model.train(X_train, y_train)

        score = scoring_model.predict(input_features)

    except Exception as e:
        logging.error(f"Model scoring failed: {e}")
        return None

    # 4. Explainability Layer
    logging.info("Step 4: Generating explanation...")
    try:
        explanation = generate_explanation(
            model=scoring_model,
            input_features=input_features,
            latest_events=unstructured_events
        )
    except Exception as e:
        logging.error(f"Explanation generation failed: {e}")
        return None

    # 5. Store Results
    logging.info("Step 5: Storing results...")
    result_record = {
        'timestamp': datetime.now().isoformat(),
        'company': company_ticker,
        'score': int(score), # <-- FIX: Convert NumPy int64 to Python int here
        'explanation': explanation,
        'raw_features': input_features.to_dict('records')[0],
        'events': unstructured_events
    }

    logging.info(json.dumps(result_record, indent=2))

    return result_record

if __name__ == '__main__':
    for ticker in COMPANIES_TO_TRACK:
        run_pipeline(ticker)
