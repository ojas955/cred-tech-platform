import yfinance as yf
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

from models.scoring_engine import CreditScoringModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# A more diverse list of companies for a robust training set
TRAINING_TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'JPM', 'V', 'JNJ', 'PG',
    'NVDA', 'HD', 'MA', 'UNH', 'BAC', 'DIS', 'PYPL', 'NFLX', 'ADBE', 'CRM'
]
TRAINING_YEARS = 5  # Use 5 years of historical data


def fetch_training_data(tickers, years):
    """Fetches historical financials and price data for multiple tickers."""
    all_features = []
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)

    for ticker in tickers:
        logging.info(f"Fetching data for {ticker}...")
        try:
            stock = yf.Ticker(ticker)

            financials = stock.financials
            balance_sheet = stock.balance_sheet

            if financials.empty or balance_sheet.empty:
                logging.warning(f"No financial data for {ticker}. Skipping.")
                continue

            price_data = stock.history(period=f'{years + 2}y', interval='1d')
            if price_data.empty:
                logging.warning(f"No price history found for {ticker}. Skipping.")
                continue
            price_data.index = price_data.index.tz_localize(None)

            for year_date_col in financials.columns:
                if year_date_col.year < start_date.year:
                    continue

                if year_date_col not in balance_sheet.columns:
                    continue

                try:
                    next_year_date = year_date_col + pd.DateOffset(years=1)
                    current_price_index = price_data.index.asof(year_date_col)
                    future_price_index = price_data.index.asof(next_year_date)

                    if pd.isna(current_price_index) or pd.isna(future_price_index):
                        continue

                    current_price = price_data.loc[current_price_index]['Close']
                    future_price = price_data.loc[future_price_index]['Close']

                    fin_data = financials[year_date_col]
                    bs_data = balance_sheet[year_date_col]

                    revenue = fin_data.get('Total Revenue')
                    op_income = fin_data.get('Operating Income')
                    net_income = fin_data.get('Net Income')

                    total_assets = bs_data.get('Total Assets')
                    current_assets = bs_data.get('Total Current Assets')
                    current_liabilities = bs_data.get('Total Current Liabilities')
                    total_debt = bs_data.get('Total Debt', 0)  # Default to 0 if not present
                    stockholder_equity = bs_data.get('Total Stockholder Equity')

                    target = 1 if future_price > current_price else 0

                    all_features.append({
                        'ticker': ticker,
                        'year': year_date_col.year,
                        'debt_to_equity': total_debt / stockholder_equity if stockholder_equity else np.nan,
                        'current_ratio': current_assets / current_liabilities if current_liabilities else np.nan,
                        'operating_margin': op_income / revenue if revenue else np.nan,
                        'return_on_assets': net_income / total_assets if total_assets else np.nan,
                        'target': target
                    })
                except Exception as e:
                    logging.warning(f"Skipping year {year_date_col.year} for {ticker} due to an error: {e}")
                    continue
        except Exception as e:
            logging.error(f"Could not process {ticker}: {e}")

    # --- DEFINITIVE FIX: Replace .dropna() with .fillna(0) ---
    df = pd.DataFrame(all_features)
    # Fill any remaining NaN values with 0 instead of dropping the row
    df.fillna(0, inplace=True)
    # Replace infinite values that can result from division by zero
    df.replace([np.inf, -np.inf], 0, inplace=True)
    return df


if __name__ == '__main__':
    logging.info("--- Starting Model Training ---")

    training_df = fetch_training_data(TRAINING_TICKERS, TRAINING_YEARS)

    if training_df.empty:
        logging.error("Failed to create a training dataset. Exiting.")
    else:
        logging.info(f"Training dataset created with {len(training_df)} records.")
        print("--- Sample of Training Data ---")
        print(training_df.head())

        features = ['debt_to_equity', 'current_ratio', 'operating_margin', 'return_on_assets']
        X_train = training_df[features]
        y_train = training_df['target']

        model = CreditScoringModel()
        model.features = features
        model.train(X_train, y_train)
        logging.info("Model training complete.")

        model.save_model('credit_scoring_model.joblib')