import yfinance as yf
import pandas as pd

def fetch_yahoo_finance_data(ticker):
    """Fetches historical financial data for a given ticker."""
    try:
        data = yf.Ticker(ticker)
        # Fetch key metrics: balance sheet, cash flow, historical prices
        balance_sheet = data.balance_sheet
        cashflow = data.cashflow
        history = data.history(period="1y")

        # Simple feature engineering (e.g., calculate debt-to-equity ratio)
        return {
            'balance_sheet': balance_sheet,
            'cashflow': cashflow,
            'history': history
        }
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

# Add more functions for other structured sources (FRED, etc.)