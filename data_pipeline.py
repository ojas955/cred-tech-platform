import pandas as pd
from datetime import datetime
import json
import logging

from data_ingestion.structured_ingestion import fetch_yahoo_finance_data
from data_ingestion.unstructured_ingestion import fetch_news_headlines, process_headlines
from models.scoring_engine import CreditScoringModel
from models.explainability_layer import generate_explanation

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

COMPANIES_TO_TRACK = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
MODEL_FILE_PATH = 'credit_scoring_model.joblib'


def run_pipeline(company_ticker: str, model: CreditScoringModel):
    logging.info(f"--- Starting pipeline for {company_ticker} ---")

    try:
        structured_data = fetch_yahoo_finance_data(company_ticker)
        if structured_data is None or 'financials' not in structured_data or structured_data['financials'].empty:
            logging.error("Failed to ingest structured data. Exiting.")
            return None

        headlines = fetch_news_headlines(company_ticker)
        unstructured_events = process_headlines(headlines)
    except Exception as e:
        logging.error(f"Data ingestion failed: {e}")
        return None

    logging.info("Preparing data for the model...")

    # --- UPDATED DATA HANDLING ---
    # Select the most recent column (which is a pandas Series)
    latest_financials = structured_data['financials'].iloc[:, 0]
    latest_balance_sheet = structured_data['balance_sheet'].iloc[:, 0]

    sentiment_score = sum(1 for e in unstructured_events if e['sentiment'] == 'positive') - \
                      sum(1 for e in unstructured_events if e['sentiment'] == 'negative')

    input_features = pd.DataFrame([{
        'debt_to_equity': latest_balance_sheet.get('Total Debt', 0) / latest_balance_sheet.get(
            'Total Stockholder Equity', 1),
        'current_ratio': latest_balance_sheet.get('Total Current Assets', 0) / latest_balance_sheet.get(
            'Total Current Liabilities', 1),
        'operating_margin': latest_financials.get('Operating Income', 0) / latest_financials.get('Total Revenue', 1),
        'return_on_assets': latest_financials.get('Net Income', 0) / latest_balance_sheet.get('Total Assets', 1)
    }]).fillna(0)

    logging.info("Running the scoring engine...")
    base_score = model.predict(input_features)

    final_score = base_score
    sentiment_adjustment_reason = "No significant sentiment impact."
    if sentiment_score <= -3:
        final_score = 0
        sentiment_adjustment_reason = "Score downgraded due to highly negative recent news."

    logging.info("Generating explanation...")
    explanation = generate_explanation(
        model=model,
        input_features=input_features,
        latest_events=unstructured_events
    )
    explanation['sentiment_adjustment'] = sentiment_adjustment_reason

    result_record = {
        'timestamp': datetime.now().isoformat(),
        'company': company_ticker,
        'base_score_financials': int(base_score),
        'final_score_with_sentiment': int(final_score),
        'explanation': explanation,
        'raw_features': input_features.to_dict('records')[0],
        'sentiment_score': sentiment_score,
        'events': unstructured_events
    }

    logging.info(f"--- RESULT FOR {company_ticker} ---")
    print(json.dumps(result_record, indent=2))
    return result_record


if __name__ == '__main__':
    logging.info(f"Loading pre-trained model from {MODEL_FILE_PATH}...")
    scoring_model = CreditScoringModel.load_model(MODEL_FILE_PATH)

    if scoring_model:
        for ticker in COMPANIES_TO_TRACK:
            run_pipeline(ticker, scoring_model)
    else:
        logging.error("Could not run pipeline because model failed to load.")