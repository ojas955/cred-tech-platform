import yfinance as yf
import logging


def fetch_yahoo_finance_data(ticker):
    """Fetches real historical financial data and balance sheet using yfinance."""
    logging.info(f"Fetching structured data for {ticker}...")
    try:
        stock = yf.Ticker(ticker)
        # Fetch financials (income statement) and balance sheet as raw DataFrames
        financials = stock.financials
        balance_sheet = stock.balance_sheet

        return {
            'financials': financials,
            'balance_sheet': balance_sheet
        }
    except Exception as e:
        logging.error(f"Error fetching real data for {ticker}: {e}")
        return None